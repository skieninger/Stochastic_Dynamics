from Code import dummy
import pytest

def test_dummy_function():
    assert dummy.dummy_function()==1

def test_sanity():
    assert 1==1
