{ SPDX-License-Identifier: AGPL-3.0-only
  CUDA single-precision intrinsics needed by the ports: fmaf (correctly rounded fused multiply-add),
  __float_as_uint. fmaf: a*b is exact in binary64; s = p + c is computed with TwoSum to get the exact
  error e, then s is rounded-to-odd (if e<>0) before the final rounding to binary32, which makes the
  result correctly rounded (53 >= 24 + 2). }
{$mode objfpc}{$Q-}{$R-}
unit SK_FloatOps;

interface

function FmaF(A, B, C: Single): Single;
function FloatAsUInt(F: Single): LongWord; inline;
function UIntAsFloat(U: LongWord): Single; inline;

implementation

uses Math;

function FmaF(A, B, C: Single): Single;
var p, s, bb, e: Double; q: QWord;
begin
  if IsNan(A) or IsNan(B) or IsNan(C) or IsInfinite(A) or IsInfinite(B) or IsInfinite(C) then
    Exit(Single(Double(A) * Double(B) + Double(C)));
  p := Double(A) * Double(B);          { exact: 24+24 <= 53 bits }
  s := p + Double(C);
  bb := s - p;
  e := (p - (s - bb)) + (Double(C) - bb);   { TwoSum: p + c = s + e exactly }
  if (e <> 0.0) and not IsInfinite(s) then
  begin
    q := PQWord(@s)^;
    if (q and 1) = 0 then
    begin
      { move s one ulp toward e: makes the last bit odd (round-to-odd) }
      if (e > 0) = (s > 0) then Inc(q) else Dec(q);
      s := PDouble(@q)^;
    end;
  end;
  Result := Single(s);
end;

function FloatAsUInt(F: Single): LongWord; inline;
begin Result := PLongWord(@F)^; end;

function UIntAsFloat(U: LongWord): Single; inline;
begin Result := PSingle(@U)^; end;

end.
