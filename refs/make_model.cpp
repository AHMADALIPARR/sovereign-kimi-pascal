// SPDX-License-Identifier: AGPL-3.0-only
// Writes a small deterministic K3M1 model (layout of k3_nano_harness.cu:576-681) and K3D1 draft (:922-960).
#include "common_ref.h"
static void wr(FILE* f,Rng& r,size_t n,float s){ for(size_t i=0;i<n;i++){ float x=r.nextf()*s; fwrite(&x,4,1,f);} }
int main(){ Rng r{99}; int dim=64,layers=2,E=4,k=2,V=260,S=16,in=48; FILE* f=fopen("model_small.bin","wb");
  fwrite("K3M1",1,4,f); int hdr[8]={1,dim,layers,E,k,V,S,in}; fwrite(hdr,4,8,f); char t[4]={0,0,0,0}; fwrite(t,1,4,f);
  wr(f,r,(size_t)V*dim,1); wr(f,r,(size_t)S*dim,0.5f); wr(f,r,dim,1); wr(f,r,(size_t)V*dim,0.3f);
  for(int l=0;l<layers;l++){ wr(f,r,dim,1); wr(f,r,3*dim*dim,0.2f); wr(f,r,dim*dim,0.2f); wr(f,r,dim,1); wr(f,r,E*dim,0.5f); wr(f,r,(size_t)E*dim*in,0.2f); wr(f,r,(size_t)E*in*dim,0.2f); }
  fclose(f);
  f=fopen("draft_small.bin","wb"); fwrite("K3D1",1,4,f); int ver=1; fwrite(&ver,4,1,f);
  int dl=1, di=32; struct { int dim,inter,ne,topk,layers,vocab,maxseq,tied; float eps; } c={dim,di,0,0,dl,V,S,0,0.0f}; fwrite(&c,sizeof(c),1,f);
  wr(f,r,dim,1); wr(f,r,(size_t)V*dim,0.3f); for(int l=0;l<dl;l++){ wr(f,r,dim,1); wr(f,r,3*dim*dim,0.2f); wr(f,r,dim*dim,0.2f); wr(f,r,dim,1); wr(f,r,di*dim,0.2f); wr(f,r,dim*di,0.2f); }
  fclose(f); }
