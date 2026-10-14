{ SPDX-License-Identifier: AGPL-3.0-only
  Port of kimi_butterfly/audit/state_audit.rs (no_std cryptographic state-event chain). }
{$mode objfpc}{$H+}{$Q-}{$R-}
unit SK_StateAudit;

interface

type
  { state_audit.rs:2 }
  TAuditEvent = record
    Node: LongWord; Epoch: QWord; Kind: Word; State: LongWord; Prev: LongWord; Seal: LongWord;
  end;

{ state_audit.rs:3 — murmur3 fmix32 with wrapping multiplies }
function Mix(X: LongWord): LongWord;
{ state_audit.rs:4 — epoch/kind are truncated/zero-extended to u32 exactly as `as u32` }
function SealEvent(const E: TAuditEvent): LongWord;
{ state_audit.rs:5-144 — audit_NNN(x) = mix(x xor K[NNN]); N in 1..140 }
function Audit(N: Integer; X: LongWord): LongWord;

implementation

{$I state_audit_table.inc}

function Mix(X: LongWord): LongWord;
begin
  X := X xor (X shr 16);
  X := LongWord(X * LongWord($85ebca6b));
  X := X xor (X shr 13);
  X := LongWord(X * LongWord($c2b2ae35));
  Result := X xor (X shr 16);
end;

function SealEvent(const E: TAuditEvent): LongWord;
begin
  Result := Mix(E.Node xor LongWord(E.Epoch) xor LongWord(E.Kind) xor E.State xor E.Prev);
end;

function Audit(N: Integer; X: LongWord): LongWord;
begin
  Assert((N >= 1) and (N <= 140), 'audit index out of range (state_audit.rs defines 001..140)');
  Result := Mix(X xor AUDIT_K[N]);
end;

end.
