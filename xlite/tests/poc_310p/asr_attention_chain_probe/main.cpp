#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_asr_attention_mmad_block_probe.h"
#include "aclrtlaunch_asr_paged_kv_stage_probe.h"

namespace {
constexpr uint32_t kHeadDim = 128, kHeads = 8, kBlockSize = 128, kTokens = 16;
constexpr uint32_t kTableStride = 16;
constexpr size_t kTileElements = kTokens * kHeadDim, kGuardElements = 16;
constexpr uint16_t kGuard = 0x7e00;
void Check(aclError e, const char *op) { if (e != ACL_SUCCESS) throw std::runtime_error(std::string(op) + " failed, aclError=" + std::to_string(e)); }
struct Buffer { void *ptr=nullptr; explicit Buffer(size_t n){Check(aclrtMalloc(&ptr,n,ACL_MEM_MALLOC_HUGE_FIRST),"aclrtMalloc");} ~Buffer(){if(ptr)(void)aclrtFree(ptr);} Buffer(const Buffer&)=delete; };
uint16_t HB(float x){const __fp16 y=static_cast<__fp16>(x);uint16_t b;std::memcpy(&b,&y,2);return b;}
float HV(uint16_t b){__fp16 x;std::memcpy(&x,&b,2);return static_cast<float>(x);}
size_t CI(uint32_t b,uint32_t t,uint32_t h,uint32_t d){return (((static_cast<size_t>(b)*kBlockSize+t)*kHeads+h)*kHeadDim+d);}
float CV(uint32_t b,uint32_t t,uint32_t h,uint32_t d,bool v){uint32_t x=b*23+t*11+h*7+d*(v?5:3);return static_cast<float>(static_cast<int32_t>(x%41)-20)/(v?64.0F:96.0F);}
void CopyH2D(Buffer &d,const void *h,size_t n,const char *op){Check(aclrtMemcpy(d.ptr,n,h,n,ACL_MEMCPY_HOST_TO_DEVICE),op);}
size_t GuardErrors(const std::vector<uint16_t>&x){size_t n=0;for(size_t i=0;i<kGuardElements;++i){n+=x[i]!=kGuard;n+=x[x.size()-1-i]!=kGuard;}return n;}

struct Metric { double cosine=0,maxAbs=0; size_t maxIndex=0,mismatches=0,nonFinite=0; };
Metric Compare(const std::vector<uint16_t>&a,const std::vector<uint16_t>&e)
{
    Metric r; double dot=0,an=0,en=0;
    for(size_t i=0;i<a.size();++i){double x=HV(a[i]),y=HV(e[i]),z=std::abs(x-y);if(z>r.maxAbs){r.maxAbs=z;r.maxIndex=i;}r.mismatches+=z>0.02;r.nonFinite+=!std::isfinite(x);dot+=x*y;an+=x*x;en+=y*y;}
    r.cosine=dot/std::sqrt(an*en);return r;
}

void Run(uint32_t start,uint32_t valid,uint32_t head,uint32_t warmup,uint32_t iterations)
{
    if(!valid||valid>16||start+valid>2048||head>=8||!iterations)throw std::invalid_argument("invalid chain case");
    constexpr uint32_t blocks=23;
    const size_t cacheN=static_cast<size_t>(blocks)*kBlockSize*kHeads*kHeadDim;
    std::vector<uint16_t> kc(cacheN),vc(cacheN); std::vector<int32_t> table(kTableStride);
    for(uint32_t l=0;l<kTableStride;++l)table[l]=static_cast<int32_t>((l*17+3)%blocks);
    for(uint32_t b=0;b<blocks;++b)for(uint32_t t=0;t<kBlockSize;++t)for(uint32_t h=0;h<kHeads;++h)for(uint32_t d=0;d<kHeadDim;++d){size_t i=CI(b,t,h,d);kc[i]=HB(CV(b,t,h,d,false));vc[i]=HB(CV(b,t,h,d,true));}
    const auto kb=kc,vb=vc;
    std::vector<uint16_t> q(16*kHeadDim),p(16*16),expectedQ(16*16),expectedP(16*kHeadDim);
    for(uint32_t r=0;r<16;++r)for(uint32_t d=0;d<kHeadDim;++d)q[r*kHeadDim+d]=HB(static_cast<float>(static_cast<int32_t>((d*3+7)%29)-14)/128.0F);
    float psum=0;for(uint32_t t=0;t<valid;++t)psum+=static_cast<float>(t+1);
    for(uint32_t r=0;r<16;++r)for(uint32_t t=0;t<16;++t)p[r*16+t]=HB(t<valid?static_cast<float>(t+1)/psum:0.0F);
    std::vector<uint16_t> kStage(kTileElements),vStage(kTileElements);
    for(uint32_t t=0;t<valid;++t){uint32_t lt=start+t,b=static_cast<uint32_t>(table[lt/128]),ib=lt%128;for(uint32_t d=0;d<128;++d){kStage[t*128+d]=kc[CI(b,ib,head,d)];vStage[d*16+t]=vc[CI(b,ib,head,d)];}}
    for(uint32_t r=0;r<16;++r)for(uint32_t n=0;n<16;++n){float s=0;for(uint32_t k=0;k<128;++k)s+=HV(q[r*128+k])*HV(kStage[n*128+k]);expectedQ[r*16+n]=HB(s);}
    for(uint32_t r=0;r<16;++r)for(uint32_t n=0;n<128;++n){float s=0;for(uint32_t k=0;k<16;++k)s+=HV(p[r*16+k])*HV(vStage[n*16+k]);expectedP[r*128+n]=HB(s);}

    std::vector<uint16_t> kScratch(kTileElements+32,kGuard),vScratch(kTileElements+32,kGuard),qOut(256+32,kGuard),pOut(2048+32,kGuard);
    Buffer kcd(kc.size()*2),vcd(vc.size()*2),td(table.size()*4),qd(q.size()*2),pd(p.size()*2);
    Buffer ksd(kScratch.size()*2),vsd(vScratch.size()*2),qod(qOut.size()*2),pod(pOut.size()*2);
    CopyH2D(kcd,kc.data(),kc.size()*2,"copy K cache");CopyH2D(vcd,vc.data(),vc.size()*2,"copy V cache");CopyH2D(td,table.data(),table.size()*4,"copy table");CopyH2D(qd,q.data(),q.size()*2,"copy Q");CopyH2D(pd,p.data(),p.size()*2,"copy P");CopyH2D(ksd,kScratch.data(),kScratch.size()*2,"copy K scratch");CopyH2D(vsd,vScratch.data(),vScratch.size()*2,"copy V scratch");CopyH2D(qod,qOut.data(),qOut.size()*2,"copy Q output");CopyH2D(pod,pOut.data(),pOut.size()*2,"copy P output");
    void *ks=static_cast<uint8_t*>(ksd.ptr)+32,*vs=static_cast<uint8_t*>(vsd.ptr)+32,*qo=static_cast<uint8_t*>(qod.ptr)+32,*po=static_cast<uint8_t*>(pod.ptr)+32;
    aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream),"create stream");
    auto chain=[&](){
        ACLRT_LAUNCH_KERNEL(asr_paged_kv_stage_probe)(1,stream,kcd.ptr,vcd.ptr,td.ptr,ks,vs,start,valid,head);
        ACLRT_LAUNCH_KERNEL(asr_attention_mmad_block_probe)(1,stream,qd.ptr,ks,qo,16,16,128);
        ACLRT_LAUNCH_KERNEL(asr_attention_mmad_block_probe)(1,stream,pd.ptr,vs,po,16,128,16);
    };
    for(uint32_t i=0;i<warmup;++i)chain();Check(aclrtSynchronizeStream(stream),"warmup sync");auto begin=std::chrono::steady_clock::now();for(uint32_t i=0;i<iterations;++i)chain();Check(aclrtSynchronizeStream(stream),"timed sync");auto end=std::chrono::steady_clock::now();
    Check(aclrtMemcpy(kScratch.data(),kScratch.size()*2,ksd.ptr,kScratch.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"copy K scratch back");Check(aclrtMemcpy(vScratch.data(),vScratch.size()*2,vsd.ptr,vScratch.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"copy V scratch back");Check(aclrtMemcpy(qOut.data(),qOut.size()*2,qod.ptr,qOut.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"copy Q output back");Check(aclrtMemcpy(pOut.data(),pOut.size()*2,pod.ptr,pOut.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"copy P output back");Check(aclrtMemcpy(kc.data(),kc.size()*2,kcd.ptr,kc.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"copy K cache back");Check(aclrtMemcpy(vc.data(),vc.size()*2,vcd.ptr,vc.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"copy V cache back");Check(aclrtDestroyStream(stream),"destroy stream");
    std::vector<uint16_t> qa(qOut.begin()+16,qOut.end()-16),pa(pOut.begin()+16,pOut.end()-16);Metric qm=Compare(qa,expectedQ),pm=Compare(pa,expectedP);
    size_t guards=GuardErrors(kScratch)+GuardErrors(vScratch)+GuardErrors(qOut)+GuardErrors(pOut);size_t cacheChanged=!std::equal(kc.begin(),kc.end(),kb.begin())+!std::equal(vc.begin(),vc.end(),vb.begin());double ms=std::chrono::duration<double,std::milli>(end-begin).count()/iterations;
    if(qm.cosine<0.999||pm.cosine<0.999||qm.maxAbs>0.02||pm.maxAbs>0.02||qm.nonFinite||pm.nonFinite||guards||cacheChanged){std::cerr<<"Attention chain diagnostics: start="<<start<<", valid="<<valid<<", head="<<head<<", qk_cos="<<qm.cosine<<", qk_max="<<qm.maxAbs<<" at "<<qm.maxIndex<<", pv_cos="<<pm.cosine<<", pv_max="<<pm.maxAbs<<" at "<<pm.maxIndex<<", guards="<<guards<<", cache_changed="<<cacheChanged<<std::endl;throw std::runtime_error("attention chain contract failed");}
    std::cout<<std::fixed<<std::setprecision(6)<<"ASR attention chain PASS: start="<<start<<", valid="<<valid<<", head="<<head<<", average_ms="<<ms<<", qk_cosine="<<qm.cosine<<", pv_cosine="<<pm.cosine<<", guards="<<guards<<", cache_changed="<<cacheChanged<<std::endl;
}
} // namespace
int main(int argc,char**argv){if(argc!=6){std::cerr<<"usage: "<<argv[0]<<" START VALID HEAD WARMUP ITERATIONS\n";return 2;}try{Check(aclInit(nullptr),"aclInit");Check(aclrtSetDevice(0),"set device");Run(std::stoul(argv[1]),std::stoul(argv[2]),std::stoul(argv[3]),std::stoul(argv[4]),std::stoul(argv[5]));Check(aclrtResetDevice(0),"reset device");Check(aclFinalize(),"aclFinalize");return 0;}catch(const std::exception&e){std::cerr<<"ASR attention chain FAILED: "<<e.what()<<std::endl;return 1;}}
