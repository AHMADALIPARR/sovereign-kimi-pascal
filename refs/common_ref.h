// SPDX-License-Identifier: AGPL-3.0-only
// Shared helpers for the CPU reference implementations derived from the .cu sources.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
struct Rng{ uint64_t s; uint64_t next(){ s+=0x9e3779b97f4a7c15ULL; uint64_t z=s; z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31);}
  float nextf(){ return (float)((int32_t)(next()>>40) - 8388608) / 8388608.0f; } };
typedef _Float16 half_t;                       // compiler's IEEE binary16 (RNE conversions)
static inline uint16_t h_bits(half_t h){ uint16_t u; memcpy(&u,&h,2); return u; }
static inline half_t f2h(float f){ return (half_t)f; }
static inline float h2f(half_t h){ return (float)h; }
static inline void outf(const char* n,int i,float v){ printf("%s %d f %.17g\n",n,i,(double)v); }
static inline void outi(const char* n,int i,long long v){ printf("%s %d i %lld\n",n,i,v); }
