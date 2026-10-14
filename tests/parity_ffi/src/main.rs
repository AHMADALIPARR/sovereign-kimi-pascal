// SPDX-License-Identifier: AGPL-3.0-only
// Parity driver for src/ffi.rs and src/daemon.rs (ORIGINAL files, via #[path]).
// The CUDA engine (k3_nano_harness.cu) cannot be built here (no nvcc/GPU), so the three
// extern "C" symbols are provided by a deterministic STUB engine; the same stub is used on the
// Pascal side. This tests the Rust host logic (wrapper, greedy sampling, channel actor), not CUDA.
#[path = "/workspace/sovereign-kimi/src/ffi.rs"] pub mod ffi;
#[path = "/workspace/sovereign-kimi/src/daemon.rs"] pub mod daemon;
use std::os::raw::c_char;
const VOCAB: usize = 300;
#[repr(C)] pub struct Stub { calls: i32 }
#[no_mangle] pub extern "C" fn k3_engine_init(_p: *const c_char) -> *mut ffi::K3EngineOpaque { Box::into_raw(Box::new(Stub{calls:0})) as *mut _ }
#[no_mangle] pub extern "C" fn k3_engine_forward(e: *mut ffi::K3EngineOpaque, token: i32, pos: i32, out: *mut f32) -> i32 {
  let s = unsafe{ &mut *(e as *mut Stub) }; s.calls += 1;
  for i in 0..VOCAB { let mut h:u32 = (token as u32).wrapping_mul(2654435761) ^ (pos as u32).wrapping_mul(40503) ^ (i as u32).wrapping_mul(2246822519);
    h ^= h>>15; h = h.wrapping_mul(0x2c1b3c6d); h ^= h>>12; unsafe{ *out.add(i) = ((h % 20001) as f32 - 10000.0) / 100.0; } }
  0 }
#[no_mangle] pub extern "C" fn k3_engine_free(e: *mut ffi::K3EngineOpaque) { unsafe{ drop(Box::from_raw(e as *mut Stub)); } }
#[tokio::main]
async fn main(){
  let eng = ffi::K3Engine::new("model.bin", VOCAB).unwrap();
  for t in 0..50 { for pos in 0..4 { let l = eng.forward(t*7+1, pos).unwrap(); println!("greedy {} {} {}", t*7+1, pos, eng.sample_greedy(&l)); } }
  println!("greedy_empty {}", eng.sample_greedy(&[]));
  println!("greedy_tie {}", eng.sample_greedy(&[1.0, 3.0, 3.0, 2.0]));
  let h = daemon::K3DaemonHandle::spawn("model.bin", VOCAB);
  println!("vocab {}", h.vocab_size());
  for t in 0..40 { let l = h.predict(t, t*3).await.unwrap(); let s: f64 = l.iter().map(|&x| x as f64).sum(); println!("predict {} {} len={} sum={:.4} first={:.2}", t, t*3, l.len(), s, l[0]); }
}
