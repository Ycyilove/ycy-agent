"""审批唯一真源：同一 approval 只能 decide 一次；verify 前必须 decide。"""
from __future__ import annotations

import time

import pytest

from tools.security.approval import ApprovalStore


def test_decide_once():
    s = ApprovalStore(ttl_seconds=60)
    rec = s.create("t1", "s1", "write_file", {"file_path": "a.txt"})
    assert s.decide(rec.approval_id, "s1", "allow") is True
    assert s.decide(rec.approval_id, "s1", "allow") is False, "重复 decide 应失败"
    assert s.decide(rec.approval_id, "s1", "deny") is False


def test_wrong_session_rejected():
    s = ApprovalStore(ttl_seconds=60)
    rec = s.create("t1", "s1", "write_file", {"file_path": "a.txt"})
    assert s.decide(rec.approval_id, "s2", "allow") is False, "session 不符应失败"
    assert s.decide(rec.approval_id, "s1", "allow") is True


def test_verify_before_decide_fails():
    s = ApprovalStore(ttl_seconds=60)
    rec = s.create("t1", "s1", "write_file", {"file_path": "a.txt"})
    assert s.verify(rec.approval_id, "write_file", {"file_path": "a.txt"}) is False


def test_verify_after_allow_passes():
    s = ApprovalStore(ttl_seconds=60)
    rec = s.create("t1", "s1", "write_file", {"file_path": "a.txt"})
    s.decide(rec.approval_id, "s1", "allow")
    assert s.verify(rec.approval_id, "write_file", {"file_path": "a.txt"}) is True


def test_verify_after_deny_fails():
    s = ApprovalStore(ttl_seconds=60)
    rec = s.create("t1", "s1", "write_file", {"file_path": "a.txt"})
    s.decide(rec.approval_id, "s1", "deny")
    assert s.verify(rec.approval_id, "write_file", {"file_path": "a.txt"}) is False


def test_verify_params_mismatch_fails():
    s = ApprovalStore(ttl_seconds=60)
    rec = s.create("t1", "s1", "write_file", {"file_path": "a.txt"})
    s.decide(rec.approval_id, "s1", "allow")
    assert s.verify(rec.approval_id, "write_file", {"file_path": "b.txt"}) is False


def test_verify_expired_fails():
    s = ApprovalStore(ttl_seconds=0)
    rec = s.create("t1", "s1", "write_file", {"file_path": "a.txt"})
    s.decide(rec.approval_id, "s1", "allow")
    time.sleep(0.01)
    assert s.verify(rec.approval_id, "write_file", {"file_path": "a.txt"}) is False


def test_allow_always_acts_as_allow():
    s = ApprovalStore(ttl_seconds=60)
    rec = s.create("t1", "s1", "write_file", {"file_path": "a.txt"})
    assert s.decide(rec.approval_id, "s1", "allow-always") is True
    assert s.verify(rec.approval_id, "write_file", {"file_path": "a.txt"}) is True