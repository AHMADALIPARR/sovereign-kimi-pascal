// SPDX-License-Identifier: AGPL-3.0-only
// CPU reference for kernels/k3_nano_harness.cu, written as straightforward sequential loops that follow
// the CUDA source line by line (thread loops in index order, warp shuffles as explicit butterflies).
// It is NOT the CUDA original: no GPU or nvcc is available here.
#include "common_ref.h"
#include <cstdlib>
#include <algorithm>
static void wred_sum(float* v){ for(int off=16;off>0;off/=2){ float t[32]; for(int l=0;l<32;l++) t[l]=v[l^off]; for(int l=0;l<32;l++) v[l]+=t[l]; } }
static float xorshift(uint64_t* s){ uint64_t x=*s; x^=x>>12; x^=x<<25; x^=x>>27; *s=x; return (float)(uint32_t)((x*0x2545F4914F6CDD1DULL)>>32)/4294967296.0f; }
// :142-164 (blockDim threads)
static void rmsnorm(const float* in,const float* w,float* out,float eps,int dim,int bd){ std::vector<float> s(bd);
  for(int t=0;t<bd;t++){ float a=0; for(int i=t;i<dim;i+=bd){ float x=in[i]; a+=x*x; } s[t]=a; }
  for(int st=bd/2;st>0;st>>=1) for(int t=0;t<st;t++) s[t]+=s[t+st];
  float rms=1.0f/sqrtf(s[0]/dim+eps); for(int i=0;i<dim;i++) out[i]=in[i]*rms*w[i]; }
static void rowdot(const float* in,const float* w,float* out,int rows,int dim){ for(int r=0;r<rows;r++){ float s=0; for(int i=0;i<dim;i++) s+=w[(size_t)r*dim+i]*in[i]; out[r]=s; } }
static void delta_attn(const float* q,const float* pk,const float* ck,const float* v,float* o,float sc,int dim){ for(int i=0;i<dim;i++){ float d=ck[i]-pk[i]; o[i]=tanhf(q[i]*d*sc)*v[i]; } }
static void router(const float* in,const float* rw,int* sel,float* rwt,int dim,int ne,int k){ std::vector<float> lg(ne);
  for(int e=0;e<ne;e++){ float s=0; for(int d=0;d<dim;d++) s+=in[d]*rw[e*dim+d]; lg[e]=s; }
  for(int j=0;j<k;j++){ float mv=-1e9f; int mi=-1; for(int e=0;e<ne;e++) if(lg[e]>mv){mv=lg[e];mi=e;} sel[j]=mi; rwt[j]=mv; if(mi!=-1) lg[mi]=-1e9f; }
  float se=0; for(int j=0;j<k;j++){ rwt[j]=expf(rwt[j]); se+=rwt[j]; } for(int j=0;j<k;j++) rwt[j]/=se; }
// :267-295
static void warp_moe(const float* in,const float* w1,const float* w2,const int* sel,const float* rwt,float* out,int dim,int inter,int k,int grid,int bd){
  for(int b=0;b<grid;b++) for(int w=0;w<bd/32;w++){ int wid=(b*bd+w*32)/32; if(wid>=dim) continue; float acc[32];
    for(int l=0;l<32;l++){ acc[l]=0; for(int kk=0;kk<k;kk++){ int ex=sel[kk]; float r=rwt[kk]; const float* a=w1+(size_t)ex*dim*inter; const float* c=w2+(size_t)ex*inter*dim;
      for(int j=l;j<inter;j+=32){ float h=0; for(int i=0;i<dim;i++) h+=in[i]*a[(size_t)i*inter+j]; h=h/(1.0f+expf(-h)); acc[l]+=h*c[(size_t)j*dim+wid]*r; } } }
    wred_sum(acc); out[wid]=acc[0]; } }
// :301-385, block-serial warps, zero-initialised shared memory
static void mma(float C[16][16],const half_t* A,int lda,const half_t* B,int ldb){ for(int m=0;m<16;m++) for(int n=0;n<16;n++){ float s=C[m][n]; for(int k=0;k<16;k++) s+=h2f(A[m*lda+k])*h2f(B[(size_t)n*ldb+k]); C[m][n]=s; } }
static void wmma_moe(const half_t* in,const half_t* w1,const half_t* w2,const int* sel,const float* rwt,float* out,int dim,int inter,int k,int dp,int ip,int grid,int bd){
  int nw=bd/32, ot=(dp+15)/16;
  for(int b=0;b<grid;b++){ std::vector<half_t> si(dp,(half_t)0.0f), sh(ip,(half_t)0.0f); half_t ib[256]={}, hb[256]={}; std::vector<int> act(nw);
    for(int w=0;w<nw;w++){ act[w]=(b*nw+w)<ot; if(!act[w]) continue; for(int l=0;l<32;l++) for(int i=w*32+l;i<dp;i+=bd) si[i]=in[i]; }
    for(int w=0;w<nw;w++){ if(!act[w]) continue; int wid=b*nw+w, os=wid*16, vo=std::min(16,dim-os); float of[16][16]={};
      for(int kk=0;kk<k;kk++){ int ex=sel[kk]; float r=rwt[kk]; const half_t* a=w1+(size_t)ex*dp*ip; const half_t* c=w2+(size_t)ex*ip*dp;
        for(int jt=0;jt<(ip+15)/16;jt++){ int js=jt*16; float hf[16][16]={};
          for(int it=0;it<(dp+15)/16;it++){ int is=it*16; for(int l=0;l<32;l++){ int row=l/16,col=l%16; ib[row*16+col]=(is+col<dp)?si[is+col]:(half_t)0.0f; } mma(hf,ib,16,a+(size_t)is*ip+js,ip); }
          for(int l=0;l<16;l++) if(js+l<inter){ float v=hf[0][l]; float s=v/(1.0f+expf(-v)); sh[js+l]=f2h(s*r); } }
        for(int jt=0;jt<(ip+15)/16;jt++){ int js=jt*16; for(int l=0;l<32;l++){ int row=l/16,col=l%16; hb[row*16+col]=(js+col<ip)?sh[js+col]:(half_t)0.0f; } mma(of,hb,16,c+(size_t)js*dp+os,dp); } }
      for(int l=0;l<32;l++) if(l<vo) out[os+l]=of[0][l]; } } }
// :391-414
static void conv(const float* in,half_t* out,int dim,int dp,int grid,int bd){ int stride=grid*bd;
  for(int idx=0;idx<stride;idx++){ for(int i=idx*4;i<dim;i+=stride*4){ float f[4]; for(int j=0;j<4;j++) f[j]=(i+j<dim)?in[i+j]:0.0f; for(int j=0;j<4;j++) if(i+j<dp) out[i+j]=f2h(f[j]); }
    for(int i=((dim+3)/4)*4+idx;i<dp;i+=stride) out[i]=(half_t)0.0f; } }
// :439-476
static int top_p(float* lg,int V,float T,float P,uint64_t seed){ float mx=-1e9f; for(int i=0;i<V;i++) mx=fmaxf(mx,lg[i]); float se=0;
  for(int i=0;i<V;i++){ lg[i]=expf((lg[i]-mx)/T); se+=lg[i]; } for(int i=0;i<V;i++) lg[i]/=se;
  float cs=0; int cut=V-1; for(int i=0;i<V;i++){ cs+=lg[i]; if(cs>=P){cut=i;break;} }
  float tm=0; for(int i=0;i<=cut;i++) tm+=lg[i]; for(int i=0;i<=cut;i++) lg[i]/=tm; for(int i=cut+1;i<V;i++) lg[i]=0;
  uint64_t x=seed; float r=xorshift(&x); cs=0; int tok=0; for(int i=0;i<=cut;i++){ cs+=lg[i]; if(r<cs){tok=i;break;} } return tok; }
// :482-533
static int spec(const float* TL,const float* DL,const int* dt,int V,int G,float T,uint64_t seed,int* acc){ uint64_t st=seed; int a=0;
  for(int i=0;i<G;i++){ int tok=dt[i]; const float* t=TL+(i+1)*V; const float* d=DL+i*V; float mt=-1e9f,md=-1e9f; for(int v=0;v<V;v++){ mt=fmaxf(mt,t[v]); md=fmaxf(md,d[v]); }
    float sT=0,sD=0; for(int v=0;v<V;v++){ sT+=expf((t[v]-mt)/T); sD+=expf((d[v]-md)/T); }
    float pt=expf((t[tok]-mt)/T)/sT, pd=expf((d[tok]-md)/T)/sD, ap=fminf(1.0f,pt/(pd+1e-10f)); float u=xorshift(&st);
    if(u<ap) acc[a++]=tok; else { float r=xorshift(&st), c=0; int s=0; for(int v=0;v<V;v++){ c+=expf((t[v]-mt)/T)/sT; if(r<c){s=v;break;} } acc[a++]=s; break; } }
  return a; }

// ---------------- host forward (:819-890) on a CPU model ----------------
struct Layer{ std::vector<float> aln,qkv,ao,mln,rt,w1,w2; std::vector<half_t> w1h,w2h; };
struct Model{ int dim,inter,ne,topk,layers,vocab,maxseq,tied; float eps=1e-6f; std::vector<float> te,pe,lnf,ow; std::vector<Layer> L; int dp,ip;
  std::vector<float> res,ao,moe,qkv,ln,logits,rw,kc,vc,dk; std::vector<int> sel; std::vector<half_t> resh; };
static void rd(FILE* f,std::vector<float>& v,size_t n){ v.assign(n,0.0f); if(n&&fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read\n");exit(1);} }
static void load(const char* p,Model& m){ FILE* f=fopen(p,"rb"); char h[40]; if(fread(h,1,40,f)!=40) exit(1); if(strncmp(h,"K3M1",4)) exit(1); int* I=(int*)(h+4);
  m.dim=I[1]; m.layers=I[2]; m.ne=I[3]; m.topk=I[4]; m.vocab=I[5]; m.maxseq=I[6]; m.inter=I[7]; m.tied=h[36];
  size_t d=m.dim,V=m.vocab,in=m.inter,E=m.ne; rd(f,m.te,V*d); rd(f,m.pe,(size_t)m.maxseq*d); rd(f,m.lnf,d); if(!m.tied) rd(f,m.ow,V*d); else m.ow.assign(V*d,0.0f);
  m.L.resize(m.layers); for(auto& l:m.L){ rd(f,l.aln,d); rd(f,l.qkv,3*d*d); rd(f,l.ao,d*d); rd(f,l.mln,d); rd(f,l.rt,E*d); rd(f,l.w1,E*d*in); rd(f,l.w2,E*in*d); } fclose(f);
  m.dp=((m.dim+15)/16)*16; m.ip=((m.inter+15)/16)*16; int dp=m.dp,ip=m.ip;
  for(auto& l:m.L){ l.w1h.assign((size_t)E*dp*ip,(half_t)0.0f); l.w2h.assign((size_t)E*ip*dp,(half_t)0.0f);
    for(size_t e=0;e<E;e++){ for(size_t i=0;i<d;i++) for(size_t j=0;j<in;j++) l.w1h[e*dp*ip+i*ip+j]=f2h(l.w1[e*d*in+i*in+j]);
      for(size_t i=0;i<in;i++) for(size_t j=0;j<d;j++) l.w2h[e*ip*dp+i*dp+j]=f2h(l.w2[e*in*d+i*d+j]); } }
  m.res.assign(d,0); m.ao.assign(d,0); m.moe.assign(d,0); m.qkv.assign(3*d,0); m.ln.assign(d,0); m.logits.assign(V,0); m.rw.assign(m.topk,0); m.sel.assign(m.topk,0); m.resh.assign(dp,(half_t)0.0f);
  size_t kv=(size_t)m.layers*m.maxseq*d; m.kc.assign(kv,0); m.vc.assign(kv,0); m.dk.assign((size_t)m.layers*d,0); }
static void forward(Model& m,int tok,int pos){ int dim=m.dim; float sc=1.0f/sqrtf(dim/32.0f);
  for(int t=0;t<256;t++) if(t<dim) m.res[t]=m.te[(size_t)tok*dim+t]+m.pe[(size_t)pos*dim+t];
  for(int l=0;l<m.layers;l++){ Layer& L=m.L[l]; rmsnorm(m.res.data(),L.aln.data(),m.ln.data(),m.eps,dim,256); rowdot(m.ln.data(),L.qkv.data(),m.qkv.data(),3*dim,dim);
    size_t off=(size_t)l*m.maxseq*dim+(size_t)pos*dim; memcpy(&m.kc[off],&m.qkv[dim],dim*4); memcpy(&m.vc[off],&m.qkv[2*dim],dim*4);
    delta_attn(m.qkv.data(),&m.dk[(size_t)l*dim],&m.kc[off],&m.vc[off],m.ao.data(),sc,dim); memcpy(&m.dk[(size_t)l*dim],&m.kc[off],dim*4);
    std::vector<float> ai=m.ao; rowdot(ai.data(),L.ao.data(),m.ao.data(),dim,dim);
    std::vector<float> h=m.res; for(int i=0;i<dim;i++) h[i]+=m.ao[i]; m.res=h;
    rmsnorm(m.res.data(),L.mln.data(),m.ln.data(),m.eps,dim,256); router(m.ln.data(),L.rt.data(),m.sel.data(),m.rw.data(),dim,m.ne,m.topk);
    int ot=(m.dp+15)/16, wb=(ot*32+127)/128; conv(m.ln.data(),m.resh.data(),dim,m.dp,(m.dp+255)/256,256);
    wmma_moe(m.resh.data(),L.w1h.data(),L.w2h.data(),m.sel.data(),m.rw.data(),m.moe.data(),dim,m.inter,m.topk,m.dp,m.ip,wb,128);
    for(int i=0;i<dim;i++) h[i]+=m.moe[i]; m.res=h; }
  rmsnorm(m.res.data(),m.lnf.data(),m.ln.data(),m.eps,dim,256); rowdot(m.ln.data(),m.ow.data(),m.logits.data(),m.vocab,dim); }


// ---------------- draft model (:922-1070) ----------------
struct Draft{ int dim,inter,ne,topk,layers,vocab,maxseq,tied; float eps; std::vector<float> lnf,ow; std::vector<Layer> L; std::vector<float> res,ln,qkv,ao,ffn,logits,kc,vc,dk; };
static void load_draft(const char* p,Draft& d){ FILE* f=fopen(p,"rb"); char mg[4]; int ver; if(fread(mg,1,4,f)!=4||fread(&ver,4,1,f)!=1) exit(1); if(strncmp(mg,"K3D1",4)) exit(1);
  if(fread(&d.dim,36,1,f)!=1) exit(1); d.eps=1e-6f; size_t D=d.dim,I=d.inter; rd(f,d.lnf,D); rd(f,d.ow,(size_t)d.vocab*D); d.L.resize(d.layers);
  for(auto& l:d.L){ rd(f,l.aln,D); rd(f,l.qkv,3*D*D); rd(f,l.ao,D*D); rd(f,l.mln,D); rd(f,l.w1,I*D); rd(f,l.w2,D*I); } fclose(f);
  d.res.assign(D,0); d.ln.assign(D,0); d.qkv.assign(3*D,0); d.ao.assign(D,0); d.ffn.assign(D,0); d.logits.assign(d.vocab,0); size_t kv=(size_t)d.layers*d.maxseq*D; d.kc.assign(kv,0); d.vc.assign(kv,0); d.dk.assign((size_t)d.layers*D,0); }
static int draft_step(Draft& d,const Model& m,int tok,int pos){ int dim=d.dim; float sc=1.0f/sqrtf(dim/32.0f);
  for(int t=0;t<256;t++) if(t<dim) d.res[t]=m.te[(size_t)tok*dim+t]+m.pe[(size_t)pos*dim+t];
  for(int l=0;l<d.layers;l++){ Layer& L=d.L[l]; rmsnorm(d.res.data(),L.aln.data(),d.ln.data(),d.eps,dim,256); rowdot(d.ln.data(),L.qkv.data(),d.qkv.data(),3*dim,dim);
    size_t off=(size_t)l*d.maxseq*dim+(size_t)pos*dim; memcpy(&d.kc[off],&d.qkv[dim],dim*4); memcpy(&d.vc[off],&d.qkv[2*dim],dim*4);
    delta_attn(d.qkv.data(),&d.dk[(size_t)l*dim],&d.kc[off],&d.vc[off],d.ao.data(),sc,dim); memcpy(&d.dk[(size_t)l*dim],&d.kc[off],dim*4);
    std::vector<float> ai=d.ao; rowdot(ai.data(),L.ao.data(),d.ao.data(),dim,dim); std::vector<float> h=d.res; for(int i=0;i<dim;i++) h[i]+=d.ao[i]; d.res=h;
    rmsnorm(d.res.data(),L.mln.data(),d.ln.data(),d.eps,dim,256); warp_moe(d.ln.data(),L.w1.data(),L.w2.data(),nullptr,nullptr,d.ffn.data(),dim,d.inter,0,(dim*32+127)/128,128);
    for(int i=0;i<dim;i++) h[i]+=d.ffn[i]; d.res=h; }
  rmsnorm(d.res.data(),d.lnf.data(),d.ln.data(),d.eps,dim,256); rowdot(d.ln.data(),d.ow.data(),d.logits.data(),d.vocab,dim);
  int best=0; for(int i=1;i<d.vocab;i++) if(d.logits[i]>d.logits[best]) best=i; return best; }

int main(){
  Rng r{2026};
  // 1 rmsnorm
  { int dim=300; std::vector<float> in(dim),w(dim),o(dim); for(auto&x:in)x=r.nextf()*3; for(auto&x:w)x=r.nextf(); rmsnorm(in.data(),w.data(),o.data(),1e-6f,dim,256); for(int i=0;i<dim;i++) outf("rmsnorm",i,o[i]); }
  // 2 qkv / 4 attn_out / 9 output_head
  { int dim=96; std::vector<float> in(dim),w(3*dim*dim),o(3*dim); for(auto&x:in)x=r.nextf(); for(auto&x:w)x=r.nextf(); rowdot(in.data(),w.data(),o.data(),3*dim,dim); for(int i=0;i<3*dim;i++) outf("qkv_proj",i,o[i]); }
  { int dim=500; std::vector<float> q(dim),pk(dim),ck(dim),v(dim),o(dim); for(auto&x:q)x=r.nextf()*4; for(auto&x:pk)x=r.nextf()*4; for(auto&x:ck)x=r.nextf()*4; for(auto&x:v)x=r.nextf();
    delta_attn(q.data(),pk.data(),ck.data(),v.data(),o.data(),0.7f,dim); for(int i=0;i<dim;i++) outf("delta_attn",i,o[i]); }
  { int dim=96; std::vector<float> in(dim),w(dim*dim),o(dim); for(auto&x:in)x=r.nextf(); for(auto&x:w)x=r.nextf(); rowdot(in.data(),w.data(),o.data(),dim,dim); for(int i=0;i<dim;i++) outf("attn_out_proj",i,o[i]); }
  // 5 router
  for(int rep=0;rep<5;rep++){ int dim=64,ne=16,k=4; std::vector<float> in(dim),w(ne*dim),rw(k); std::vector<int> sel(k); for(auto&x:in)x=r.nextf(); for(auto&x:w)x=r.nextf()*0.5f;
    router(in.data(),w.data(),sel.data(),rw.data(),dim,ne,k); for(int j=0;j<k;j++){ outi("router_sel",rep*k+j,sel[j]); outf("router_w",rep*k+j,rw[j]); } }
  // 6 warp moe
  { int dim=40,inter=70,E=3,k=2; std::vector<float> in(dim),w1(E*dim*inter),w2(E*inter*dim),o(dim,0); for(auto&x:in)x=r.nextf(); for(auto&x:w1)x=r.nextf()*0.3f; for(auto&x:w2)x=r.nextf()*0.3f;
    int sel[2]={2,0}; float rw[2]={0.6f,0.4f}; int g=(dim*32+127)/128; warp_moe(in.data(),w1.data(),w2.data(),sel,rw,o.data(),dim,inter,k,g,128); for(int i=0;i<dim;i++) outf("warp_moe",i,o[i]); }
  // 7 wmma moe, two shapes (second exercises the early-exit warps of the last block)
  for(int shape=0;shape<2;shape++){ int dim=shape?72:40, inter=70, E=3, k=2, dp=((dim+15)/16)*16, ip=((inter+15)/16)*16;
    std::vector<half_t> in(dp),w1((size_t)E*dp*ip),w2((size_t)E*ip*dp); for(auto&x:in)x=f2h(r.nextf()); for(auto&x:w1)x=f2h(r.nextf()*0.3f); for(auto&x:w2)x=f2h(r.nextf()*0.3f);
    std::vector<float> o(dim,-7.0f); int sel[2]={1,2}; float rw[2]={0.7f,0.3f}; int ot=(dp+15)/16, wb=(ot*32+127)/128;
    wmma_moe(in.data(),w1.data(),w2.data(),sel,rw,o.data(),dim,inter,k,dp,ip,wb,128); for(int i=0;i<dim;i++) outf(shape?"wmma_moe_72":"wmma_moe_40",i,o[i]); }
  // 8 convert
  { int dim=37,dp=48; std::vector<float> in(dim); for(auto&x:in)x=r.nextf()*70000.0f; in[3]=1e-6f; in[4]=6.1e-5f; in[5]=-0.0f; std::vector<half_t> o(dp,(half_t)1.0f);
    conv(in.data(),o.data(),dim,dp,1,256); for(int i=0;i<dp;i++) outi("fp32_to_fp16",i,h_bits(o[i])); }
  // half conversion sweep (RNE), shared helper check
  for(int i=0;i<4000;i++){ uint32_t u=(uint32_t)r.next(); float f; memcpy(&f,&u,4); if(std::isnan(f)) continue; outi("f2h_sweep",i,h_bits(f2h(f))); }
  { int V=300; std::vector<float> lg(V); for(auto&x:lg)x=r.nextf()*5; std::vector<float> o(V); rowdot(lg.data(),lg.data(),o.data(),0,0); int dim=64; std::vector<float> in(dim),w(V*dim),lo(V); for(auto&x:in)x=r.nextf(); for(auto&x:w)x=r.nextf();
    rowdot(in.data(),w.data(),lo.data(),V,dim); for(int i=0;i<V;i++) outf("output_head",i,lo[i]); }
  // 10 top-p
  for(int rep=0;rep<20;rep++){ int V=300; std::vector<float> lg(V); for(auto&x:lg)x=r.nextf()*4; uint64_t seed=r.next(); int t=top_p(lg.data(),V,0.8f,0.9f,seed); outi("top_p_token",rep,t); outf("top_p_p0",rep,lg[0]); }
  // 11 speculative acceptance
  for(int rep=0;rep<20;rep++){ int V=200,G=4; std::vector<float> TL((G+1)*V),DL(G*V); for(auto&x:TL)x=r.nextf()*3; for(auto&x:DL)x=r.nextf()*3; int dt[4]; for(int i=0;i<4;i++) dt[i]=(int)(r.next()%V);
    uint64_t seed=r.next(); int acc[4]={-1,-1,-1,-1}; int a=spec(TL.data(),DL.data(),dt,V,G,0.8f,seed,acc); outi("spec_count",rep,a); for(int i=0;i<a;i++) outi("spec_tok",rep*4+i,acc[i]); }
  // 12 embeddings
  { int dim=50,V=20,S=10,B=3; std::vector<float> te(V*dim),pe(S*dim),res(B*dim,0); for(auto&x:te)x=r.nextf(); for(auto&x:pe)x=r.nextf(); int ids[3]={4,19,0};
    for(int p=0;p<B;p++) for(int i=0;i<dim;i++) res[p*dim+i]=te[ids[p]*dim+i]+pe[(2+p)*dim+i]; for(int i=0;i<B*dim;i++) outf("batched_embedding",i,res[i]); }
  // full forward_token on tests/model_small.bin
  { Model m; load("model_small.bin",m); const char* prompt="Hi!"; int toks[8]; int n=0; toks[n++]=1; for(const char* c=prompt;*c;c++) toks[n++]=4+(unsigned char)*c;
    for(int i=0;i<n;i++) forward(m,toks[i],i); int tok=1; int idx=0;
    for(int step=0;step<4;step++){ forward(m,tok,n+step); for(int v=0;v<m.vocab;v++) outf("forward_logits",idx++,m.logits[v]);
      std::vector<float> lg=m.logits; tok=top_p(lg.data(),m.vocab,0.8f,0.9f,1000+step); outi("forward_token",step,tok); } }
  { Model m; load("model_small.bin",m); Draft d; load_draft("draft_small.bin",d); int tok=1, idx=0;
    for(int pos=0;pos<5;pos++){ tok=draft_step(d,m,tok,pos); outi("draft_best",pos,tok); for(int v=0;v<d.vocab;v++) outf("draft_logits",idx++,d.logits[v]); } }
}
