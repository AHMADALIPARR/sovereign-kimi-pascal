{ SPDX-License-Identifier: AGPL-3.0-only — output lines "name idx f|i value" read by refs/compare.cpp }
{$mode objfpc}{$H+}
unit SKOut;
interface
procedure OutF(const N: AnsiString; I: Integer; V: Single);
procedure OutI(const N: AnsiString; I: Integer; V: Int64);
implementation
uses SysUtils, Math;
procedure OutF(const N: AnsiString; I: Integer; V: Single);
var s: AnsiString;
begin
  if IsNan(V) then s := 'nan' else if IsInfinite(V) then begin if V > 0 then s := 'inf' else s := '-inf'; end
  else s := FloatToStrF(Double(V), ffExponent, 17, 3, DefaultFormatSettings);
  WriteLn(N, ' ', I, ' f ', s);
end;
procedure OutI(const N: AnsiString; I: Integer; V: Int64);
begin WriteLn(N, ' ', I, ' i ', V); end;
initialization
  DefaultFormatSettings.DecimalSeparator := '.';
end.
