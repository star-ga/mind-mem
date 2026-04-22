from mind_mem.iterative_recall import iterative_retrieve
from mind_mem.chain_of_note import chain_of_note_pack

def mock_retrieve(q):
    return [{"_id": "1", "excerpt": "test"}]

def mock_llm(p):
    return '{"followups": []}'

print("Testing iterative_retrieve with braces...")
try:
    iterative_retrieve("Question with {braces}", mock_retrieve, mock_llm)
    print("iterative_retrieve: SUCCESS")
except KeyError as e:
    print(f"iterative_retrieve: FAILED with KeyError: {e}")
except Exception as e:
    print(f"iterative_retrieve: FAILED with {type(e).__name__}: {e}")

print("\nTesting chain_of_note_pack with braces...")
try:
    chain_of_note_pack("Question with {braces}", [{"excerpt": "test"}], mock_llm)
    print("chain_of_note_pack: SUCCESS")
except KeyError as e:
    print(f"chain_of_note_pack: FAILED with KeyError: {e}")
except Exception as e:
    print(f"chain_of_note_pack: FAILED with {type(e).__name__}: {e}")
