// SPDX-License-Identifier: AGPL-3.0-only
// CPU reference derived from kimi_butterfly/cuda/kimi_butterfly.cu, kimi_butterfly/circuits/butterfly_factors.cu,
// kimi_butterfly/sass/butterfly.sass, kernels/fused_attention.cu and kernels/gemm_online_softmax.cu.
// Straightforward sequential loops; warp ops as explicit lane arrays. NOT executed on a GPU.
// Same documented choices as the port for undefined behaviour: zeroed shared memory, cp.async lands at
// wait, OOB row_max/row_denom reads -> NaN, O_acc starts at 0.
#include "common_ref.h"
#include <cmath>
#include <algorithm>
typedef float W32[32];
static void shadd(float* v,int m){ float t[32]; for(int l=0;l<32;l++) t[l]=v[l^m]; for(int l=0;l<32;l++) v[l]+=t[l]; }
static void bsum(float* v){ for(int m=16;m;m>>=1) shadd(v,m); }
static void factor_bfly(float* v){ for(int m=1;m<32;m<<=1) shadd(v,m); }
static void reduce16(float* v){ shadd(v,16);shadd(v,8);shadd(v,4);shadd(v,2);shadd(v,1); }
static void rec_reduce(float* v,int st){ if(st<=0) return; shadd(v,1<<((st-1)&4)); rec_reduce(v,st-1); }
static uint32_t fau(float f){ uint32_t u; memcpy(&u,&f,4); return u; } static float uaf(uint32_t u){ float f; memcpy(&f,&u,4); return f; }
int main(){
  Rng r{4242};
  // T1 threadpiper_router (kimi_butterfly.cu:11-18), grid 1, block 896
  { int dim=100; std::vector<float> x(dim),rw(896*dim); for(auto&v:x)v=r.nextf(); for(auto&v:rw)v=r.nextf(); int ids[16]; float w[16];
    for(int wp=0;wp<896/32;wp++){ float s[32]; for(int l=0;l<32;l++){ int e=wp*32+l; s[l]=0; for(int d=0;d<dim;d+=32){ int ix=(d+e)%dim; s[l]=fmaf(x[ix],rw[e*dim+ix],s[l]); } }
      bsum(s); for(int l=0;l<32;l++){ int t=wp*32+l; if(t<16){ ids[t]=t; w[t]=s[l]; } } }
    for(int i=0;i<16;i++){ outi("tp_router_ids",i,ids[i]); outf("tp_router_w",i,w[i]); } }
  // T2 gated_delta_butterfly (:19-25), grid 2 block 128, dim 256
  { int dim=256; std::vector<float> x(dim),p(dim),g(dim),st(dim); for(auto&v:x)v=r.nextf(); for(auto&v:p)v=r.nextf(); for(auto&v:g)v=r.nextf(); int ids[16]; float w[16];
    for(int i=0;i<16;i++){ ids[i]=(int)(r.next()%16); w[i]=r.nextf(); }
    for(int b=0;b<2;b++) for(int wp=0;wp<4;wp++){ float a[32]; for(int l=0;l<32;l++){ int i=b*128+wp*32+l, tx=wp*32+l; float d=x[i]-p[i]; a[l]=d*g[i]; a[l]*=w[ids[tx%16]]; } bsum(a); for(int l=0;l<32;l++) st[b*128+wp*32+l]=a[l]; }
    for(int i=0;i<dim;i++) outf("gated_delta",i,st[i]); }
  // T3 triple_lock_kernel (:26-31)
  { int n=1000; std::vector<float> s(n); std::vector<uint32_t> seal(n); for(int i=0;i<n;i++){ s[i]=r.nextf()*1000; seal[i]=(uint32_t)r.next(); } s[0]=-0.0f; s[1]=INFINITY; s[2]=1e-40f;
    for(int i=0;i<n;i++){ uint32_t x=fau(s[i]); uint32_t a=x^seal[i]; uint32_t b=(a<<13)|(a>>19); uint32_t c=b^(x*0x9e3779b9u); outi("triple_lock",i,c); } }
  // T4 butterfly_factors device functions
  for(int t=0;t<40;t++){ float v[32],v2[32],v3[32]; for(int l=0;l<32;l++){ v[l]=r.nextf()*10; v2[l]=v[l]; v3[l]=v[l]; } factor_bfly(v); reduce16(v2); int st=(int)(r.next()%7); rec_reduce(v3,st);
    for(int l=0;l<32;l++){ outf("factor_bfly",t*32+l,v[l]); outf("reduce16",t*32+l,v2[l]); outf("recursive_reduce",t*32+l,v3[l]); } }
  for(int t=0;t<200;t++){ float a=r.nextf()*5,b=r.nextf()*5,c=r.nextf()*5; uint32_t x=(uint32_t)r.next(),y=(uint32_t)r.next(),z=(uint32_t)r.next();
    outf("factor_gate",t,a*b); outf("factor_delta",t,a-b); outf("factor_recur",t,c+(a*b)); outi("factor_lock",t,(x^y)^z); outf("route",t,a*b); }
  for(int t=0;t<20000;t++){ uint32_t ux=(uint32_t)r.next(), uy=(uint32_t)r.next(); float x=uaf((ux&0x807FFFFFu)|((uint32_t)(100+(ux>>23)%56)<<23)), y=uaf((uy&0x807FFFFFu)|((uint32_t)(100+(uy>>23)%56)<<23));
    if(t%4==0) y=-1.0f+uaf(uy&0x0000FFFFu|0x33800000u); outf("factor_fmaf",t,fmaf(x,y,x)); }
  // T5 butterfly.sass, grid 2, block 64 (4 warps)
  { const int NW=4; uint32_t R[NW][32][32]={}; std::vector<uint32_t> mem(256); for(auto&m:mem)m=(uint32_t)r.next();
    std::vector<int> perm(256); for(int i=0;i<256;i++) perm[i]=i; for(int i=255;i>0;i--){ int j=(int)(r.next()%(i+1)); std::swap(perm[i],perm[j]); }
    for(int w=0;w<NW;w++) for(int l=0;l<32;l++){ for(int q:{6,7,8,9,11}) R[w][l][q]=fau(r.nextf()*8); R[w][l][14]=(uint32_t)r.next(); R[w][l][15]=(uint32_t)r.next(); R[w][l][20]=(uint32_t)(r.next()%256); R[w][l][21]=perm[w*32+l]; }
    for(int w=0;w<NW;w++){ int blk=w/2, tb=(w%2)*32; auto& W=R[w];
      for(int l=0;l<32;l++){ W[l][0]=tb+l; W[l][1]=blk; W[l][2]=W[l][0]^W[l][1]^0; }
      for(int m=16;m>=1;m>>=1){ uint32_t t[32]; for(int l=0;l<32;l++) t[l]=W[l^m][8]; for(int l=0;l<32;l++) W[l][8]=t[l]; for(int l=0;l<32;l++) W[l][9]=fau(uaf(W[l][9])+uaf(W[l][8])); }
      for(int l=0;l<32;l++){ W[l][10]=fau(uaf(W[l][6])+(-uaf(W[l][7]))); W[l][10]=fau(uaf(W[l][10])*uaf(W[l][11])); W[l][12]=mem[W[l][20]]; }
      for(int l=0;l<32;l++) mem[W[l][21]]=W[l][12];
      for(int l=0;l<32;l++) W[l][13]=W[l][12]^W[l][14]^W[l][15]; }
    int k=0; for(int w=0;w<NW;w++) for(int l=0;l<32;l++){ for(int q:{0,1,2,8,12,13}) outi("sass_int",k++,R[w][l][q]); outf("sass_R9",w*32+l,uaf(R[w][l][9])); outf("sass_R10",w*32+l,uaf(R[w][l][10])); }
    for(int i=0;i<256;i++) outi("sass_mem",i,mem[i]); }
  // T6 fused_attention.cu
  { const int seq=256, D=64, TM=128,TN=64,TK=64,RQ=72,RK=72; std::vector<half_t> Q(seq*D),K(seq*D); for(auto&v:Q)v=f2h(r.nextf()*2); for(auto&v:K)v=f2h(r.nextf()*2);
    std::vector<float> O(seq*D,-1.0f); const float SF=1.0f/64.0f; size_t smB=(TM*RQ+TN*RK)*2+TM*TN*4;
    for(int bx=0;bx<seq/TM;bx++){ std::vector<uint8_t> sm(smB,0); half_t* sQ=(half_t*)sm.data(); half_t* sK=sQ+TM*RQ; float mi[128],di[128],fc[128][4];
      for(int t=0;t<128;t++){ mi[t]=-INFINITY; di[t]=0; for(int f=0;f<4;f++) fc[t][f]=0; }
      int nk=seq/TK; for(int t=0;t<128;t++){ int qr=bx*TM+t/8, qc=(t%8)*8; memcpy(&sQ[(t/8)*RQ+qc],&Q[(size_t)qr*D+qc],16); }
      for(int ks=0;ks<nk;ks++){ int nx=ks+1;
        for(int m=0;m<TM;m+=16) for(int n=0;n<TN;n+=8) for(int w=0;w<4;w++){ const half_t* a=sQ+(m+w*16)*RQ; float A[16][16],B[16][8],Dm[16][8];
          for(int rr=0;rr<16;rr++) for(int c=0;c<16;c++) A[rr][c]=h2f(a[((rr>=8)+2*(c>=8))*2+(c&1)]);
          for(int l=0;l<32;l++){ int g=l>>2,tg=l&3; const half_t* b=sK+(n+l)*RK; B[2*tg][g]=h2f(b[0]); B[2*tg+1][g]=h2f(b[1]); B[2*tg+8][g]=h2f(b[2]); B[2*tg+9][g]=h2f(b[3]); }
          for(int l=0;l<32;l++){ int g=l>>2,tg=l&3,t=w*32+l; Dm[g][2*tg]=fc[t][0]; Dm[g][2*tg+1]=fc[t][1]; Dm[g+8][2*tg]=fc[t][2]; Dm[g+8][2*tg+1]=fc[t][3]; }
          for(int rr=0;rr<16;rr++) for(int c=0;c<8;c++){ float s=Dm[rr][c]; for(int kk=0;kk<16;kk++) s+=A[rr][kk]*B[kk][c]; Dm[rr][c]=s; }
          for(int l=0;l<32;l++){ int g=l>>2,tg=l&3,t=w*32+l; fc[t][0]=Dm[g][2*tg]; fc[t][1]=Dm[g][2*tg+1]; fc[t][2]=Dm[g+8][2*tg]; fc[t][3]=Dm[g+8][2*tg+1];
            for(int f=0;f<4;f++){ float mp=mi[t]; mi[t]=(fc[t][f]>mi[t])?fc[t][f]:mi[t]; float ep=expf((mp-mi[t])*SF), ec=expf((fc[t][f]-mi[t])*SF); di[t]=di[t]*ep+ec; } } }
        if(nx<nk) for(int t=0;t<128;t++){ int kr=nx*TK+t/8, kc=(t%8)*8; memcpy(&sK[(t/8)*RK+kc],&K[(size_t)kr*D+kc],16); } }
      for(int w=0;w<4;w++){ float red[32]; for(int l=0;l<32;l++) red[l]=mi[w*32+l]; for(int off=16;off>=1;off/=2){ float t[32]; for(int l=0;l<32;l++) t[l]=red[l^off]; for(int l=0;l<32;l++) red[l]=(red[l]>t[l])?red[l]:t[l]; }
        for(int l=0;l<32;l++){ int t=w*32+l; for(int f=0;f<4;f++) fc[t][f]=expf((fc[t][f]-red[l])*SF)/di[t]; int orow=bx*TM+w*16+l/4, oc=(l%4)*2;
          if(orow<seq&&oc<D){ size_t ix=((size_t)orow*(D/2)+oc/2)*2; O[ix]=h2f(f2h(fc[t][0])); O[ix+1]=h2f(f2h(fc[t][1])); } } } }
    for(int i=0;i<seq*D;i++) outf("fused_attention",i,O[i]); }
  // T7 gemm_online_softmax.cu
  { const int seq=200, HD=64, TM=128,TN=64,TK=64,RQ=72,RK=72,RPW=32; std::vector<half_t> Q(seq*HD),K(seq*HD),V(seq*HD); for(auto&v:Q)v=f2h(r.nextf()); for(auto&v:K)v=f2h(r.nextf()); for(auto&v:V)v=f2h(r.nextf());
    std::vector<float> O(seq*HD,-1.0f); float scale=1.0f/sqrtf((float)HD);
    for(int bx=0;bx<(seq+TM-1)/TM;bx++){ int qb=bx*TM; std::vector<half_t> Qs(TM*RQ,(half_t)0.0f),Ks(TN*RK,(half_t)0.0f),Vs(TN*HD,(half_t)0.0f);
      static float rmx[128][32],rdn[128][32],OA[128][32][2]; for(int t=0;t<128;t++) for(int i=0;i<32;i++){ rmx[t][i]=-INFINITY; rdn[t][i]=0; OA[t][i][0]=OA[t][i][1]=0; }
      auto RM=[&](int t,int i){ return (i>=0&&i<32)?rmx[t][i]:NAN; }; auto RD=[&](int t,int i){ return (i>=0&&i<32)?rdn[t][i]:NAN; };
      int nk=(seq+TN-1)/TN; for(int kt=0;kt<nk;kt++){ int kb=kt*TN;
        for(int i=0;i<TN*RK;i++){ int row=i/RK,col=i%RK,gr=kb+row; Ks[row*RK+col]=(row<TN&&col<TK&&gr<seq&&col<HD)?K[(size_t)gr*HD+col]:(half_t)0.0f; }
        for(int i=0;i<TM*RQ;i++){ int row=i/RQ,col=i%RQ,gr=qb+row; Qs[row*RQ+col]=(row<TM&&col<TK&&gr<seq&&col<HD)?Q[(size_t)gr*HD+col]:(half_t)0.0f; }
        for(int w=0;w<4;w++){ int ws=w*RPW; for(int rr=0;rr<RPW;rr+=16){ int gr=ws+rr; if(gr>=TM) continue; float lm[32],ls[32];
          for(int l=0;l<32;l++){ lm[l]=-INFINITY; ls[l]=0; for(int n=l;n<TN;n+=32){ float s=0; for(int k=0;k<TK;k++) s+=h2f(Qs[gr*RQ+k])*h2f(Ks[n*RK+k]); s*=scale; float pm=lm[l]; lm[l]=fmaxf(lm[l],s); ls[l]=ls[l]*expf(pm-lm[l])+expf(s-lm[l]); } }
          for(int off=16;off>0;off>>=1){ float t[32]; for(int l=0;l<32;l++) t[l]=lm[l^off]; for(int l=0;l<32;l++) lm[l]=fmaxf(lm[l],t[l]); }
          for(int off=16;off>0;off>>=1){ float t[32]; for(int l=0;l<32;l++) t[l]=ls[l^off]; for(int l=0;l<32;l++) ls[l]+=t[l]; }
          for(int l=0;l<32;l++){ int t=w*32+l; float mp=rmx[t][rr]; rmx[t][rr]=fmaxf(rmx[t][rr],lm[l]); rdn[t][rr]=rdn[t][rr]*expf(mp-rmx[t][rr])+ls[l]*expf(lm[l]-rmx[t][rr]); } } }
        for(int i=0;i<TN*HD;i++){ int row=i/HD,col=i%HD,gr=kb+row; if(gr<seq&&col<HD) Vs[row*HD+col]=V[(size_t)gr*HD+col]; }
        for(int w=0;w<4;w++){ int ws=w*RPW; for(int rr=ws;rr<ws+RPW&&rr<TM;rr++) for(int l=0;l<32;l++){ int t=w*32+l; for(int d=l;d<HD;d+=32){ float ps=0; float rm=RM(t,rr);
          for(int n=0;n<TN;n++){ float s=h2f(Qs[rr*RQ])*h2f(Ks[n*RK]); float p=expf(s*scale-rm); ps+=p*h2f(Vs[n*HD+d]); } OA[t][rr%RPW][d%2]+=ps/RD(t,rr); } } } }
      for(int w=0;w<4;w++) for(int l=0;l<32;l++){ int t=w*32+l; for(int rr=0;rr<RPW;rr++){ int gr=qb+w*RPW+rr; if(gr>=seq) continue; for(int d=l;d<HD;d+=32) O[(size_t)gr*HD+d]=OA[t][rr][d%2]/rdn[t][rr]; } } }
    for(int i=0;i<seq*HD;i++) outf("gemm_online_softmax",i,O[i]); }
}
