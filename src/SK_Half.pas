{ SPDX-License-Identifier: AGPL-3.0-only
  IEEE-754 binary16 <-> binary32 conversion used for CUDA `half`
  (__float2half / __float2half_rn = round-to-nearest-even, __half2float = exact). }
{$mode objfpc}{$Q-}{$R-}
unit SK_Half;

interface

type
  THalf = Word;
  PHalf = ^THalf;

function FloatToHalf(F: Single): THalf;
function HalfToFloat(H: THalf): Single;

implementation

function FloatToHalf(F: Single): THalf;
var x, sign, mant, rem, half_: LongWord; e: LongInt; r: LongWord;
begin
  x := PLongWord(@F)^;
  sign := (x shr 16) and $8000;
  e := LongInt((x shr 23) and $FF);
  mant := x and $7FFFFF;
  if e = 255 then
  begin
    if mant <> 0 then Exit(THalf(sign or $7E00 or (mant shr 13)));   { NaN, keep quiet bit }
    Exit(THalf(sign or $7C00));
  end;
  e := e - 127 + 15;
  if e >= 31 then Exit(THalf(sign or $7C00));  { overflow -> inf under RNE }
  if e <= 0 then
  begin
    if e < -10 then Exit(THalf(sign));          { underflow to signed zero }
    mant := mant or $800000;
    r := LongWord(14 - e);                      { shift amount 14..24 }
    half_ := 1 shl (r - 1);
    rem := mant and ((1 shl r) - 1);
    mant := mant shr r;
    if (rem > half_) or ((rem = half_) and ((mant and 1) <> 0)) then Inc(mant);
    Exit(THalf(sign or mant));                  { carry into exponent is correct encoding }
  end;
  rem := mant and $1FFF;
  mant := mant shr 13;
  r := (LongWord(e) shl 10) or mant;
  if (rem > $1000) or ((rem = $1000) and ((r and 1) <> 0)) then Inc(r);  { may carry to inf: correct }
  Result := THalf(sign or r);
end;

function HalfToFloat(H: THalf): Single;
var sign, e, mant, x: LongWord;
begin
  sign := LongWord(H and $8000) shl 16;
  e := (H shr 10) and $1F;
  mant := H and $3FF;
  if e = 0 then
  begin
    if mant = 0 then x := sign
    else
    begin
      e := 127 - 15 + 1;
      while (mant and $400) = 0 do begin mant := mant shl 1; Dec(e); end;
      mant := mant and $3FF;
      x := sign or (e shl 23) or (mant shl 13);
    end;
  end
  else if e = 31 then x := sign or $7F800000 or (mant shl 13)
  else x := sign or ((e - 15 + 127) shl 23) or (mant shl 13);
  Result := PSingle(@x)^;
end;

end.
