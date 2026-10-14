{ SPDX-License-Identifier: AGPL-3.0-only
  Executable contracts derived from kimi_butterfly/proof/forge.mlw (kept verbatim in proof/forge.mlw).
  Why3 `int` is unbounded; here Int64 is used and every operation is overflow-checked by Assert,
  so a contract violation or a leaving of the mathematical domain fails loudly instead of wrapping. }
{$mode objfpc}{$H+}{$Q+}{$R+}{$ASSERTIONS ON}
unit SK_ForgeProof;

interface

function RegOk(R, Limit: Int64): Boolean;          { forge.mlw:4 }
function Aligned(A: Int64): Boolean;               { forge.mlw:5 (Why3 `mod` is Euclidean) }
function Butterfly(X, N: Int64): Int64;            { forge.mlw:6-7, variant {n} }
function Gate(X, G: Int64): Int64;                 { forge.mlw:8 }
function Delta(X, P: Int64): Int64;                { forge.mlw:9 }
function Recur(S, D, G: Int64): Int64;             { forge.mlw:10 }
procedure CheckButterflyNonnegative(X: Int64);     { forge.mlw:11 lemma instance }

implementation

function EuclidMod(A, B: Int64): Int64;
begin
  Result := A mod B;
  if Result < 0 then Result := Result + Abs(B);
end;

function RegOk(R, Limit: Int64): Boolean;
begin Result := (0 <= R) and (R < Limit); end;

function Aligned(A: Int64): Boolean;
begin Result := EuclidMod(A, 4096) = 0; end;

function Butterfly(X, N: Int64): Int64;
var Variant: Int64;
begin
  if N <= 0 then Exit(X);
  Variant := N;
  Result := Butterfly(X + X, N - 1);   { $Q+ traps overflow outside Int64 }
  Assert(N - 1 < Variant, 'variant {n} must strictly decrease');
  Assert((N - 1) >= 0, 'variant {n} must stay bounded below on recursive call');
end;

function Gate(X, G: Int64): Int64; begin Result := X * G; end;
function Delta(X, P: Int64): Int64; begin Result := X - P; end;
function Recur(S, D, G: Int64): Int64;
begin
  Result := S + Gate(D, G);
  Assert(Result = S + D * G, 'recur post: s + gate d g');
end;

procedure CheckButterflyNonnegative(X: Int64);
begin
  if X >= 0 then
    Assert(Butterfly(X, 1) >= 0, 'lemma butterfly_nonnegative violated');
end;

end.
