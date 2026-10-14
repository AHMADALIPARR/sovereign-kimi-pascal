// SPDX-License-Identifier: AGPL-3.0-only
// compare <ref.txt> <port.txt> <abs_tol> <rel_tol>
// Lines: "<name> <idx> f|i <value>". Integers must match exactly; floats must agree within
// |a-b| <= abs_tol + rel_tol*|ref|, NaN must match NaN, +/-inf must match exactly. Reports per test name.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
struct V{ char t; double v; long long i; };
static std::map<std::pair<std::string,int>,V> load(const char* p){ std::map<std::pair<std::string,int>,V> m; std::ifstream f(p); std::string line;
  while(std::getline(f,line)){ std::istringstream s(line); std::string n,t,val; int idx; if(!(s>>n>>idx>>t>>val)) continue; V v{t[0],0,0};
    if(t=="i") v.i=atoll(val.c_str()); else v.v=strtod(val.c_str(),nullptr); m[{n,idx}]=v; } return m; }
int main(int c,char**a){ auto R=load(a[1]), P=load(a[2]); double at=atof(a[3]), rt=atof(a[4]);
  struct S{ long n=0,fail=0,miss=0; double maxabs=0,maxrel=0; char t='?'; }; std::map<std::string,S> st; long tf=0;
  for(auto& [k,rv]:R){ auto& s=st[k.first]; s.t=rv.t; s.n++; auto it=P.find(k); if(it==P.end()){ s.miss++; s.fail++; continue; } auto pv=it->second;
    if(rv.t=='i'){ if(pv.t!='i'||pv.i!=rv.i) s.fail++; continue; }
    double x=rv.v,y=pv.v; if(std::isnan(x)||std::isnan(y)){ if(!(std::isnan(x)&&std::isnan(y))) s.fail++; continue; }
    if(std::isinf(x)||std::isinf(y)){ if(x!=y) s.fail++; continue; }
    double d=fabs(x-y), rel=d/(fabs(x)>1e-30?fabs(x):1e-30); if(d>s.maxabs) s.maxabs=d; if(fabs(x)>1e-6 && rel>s.maxrel) s.maxrel=rel; if(d>at+rt*fabs(x)) s.fail++; }
  for(auto& [k,pv]:P) if(!R.count(k)){ st[k.first].n++; st[k.first].fail++; st[k.first].miss++; }
  printf("%-22s %4s %6s %6s %6s %12s %12s\n","test","type","count","fail","extra/missing","max_abs_err","max_rel_err");
  for(auto& [n,s]:st){ printf("%-22s %4c %6ld %6ld %6ld %12.3e %12.3e  %s\n",n.c_str(),s.t,s.n,s.fail,s.miss,s.maxabs,s.maxrel,s.fail?"FAIL":"PASS"); tf+=s.fail; }
  printf("TOTAL mismatches: %ld (tolerance: abs %.1e + rel %.1e * |ref|, integers exact)\n",tf,at,rt); return tf?1:0; }
