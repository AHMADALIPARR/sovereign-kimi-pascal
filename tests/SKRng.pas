{ SPDX-License-Identifier: AGPL-3.0-only  — splitmix64 shared by all parity drivers }
{$mode objfpc}{$modeswitch advancedrecords}{$Q-}{$R-}
unit SKRng;
interface
type TRng = record S: QWord; function Next: QWord; function NextF: Single; end;
function MakeRng(Seed: QWord): TRng;
implementation
function MakeRng(Seed: QWord): TRng; begin Result.S := Seed; end;
function TRng.Next: QWord;
var z: QWord;
begin
  S := S + QWord($9e3779b97f4a7c15); z := S;
  z := (z xor (z shr 30)) * QWord($bf58476d1ce4e5b9);
  z := (z xor (z shr 27)) * QWord($94d049bb133111eb);
  Result := z xor (z shr 31);
end;
{ uniform in [-1,1): top 24 bits }
function TRng.NextF: Single;
begin Result := Single(LongInt(Next shr 40) - 8388608) / 8388608.0; end;
end.
