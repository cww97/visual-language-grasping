def test_main():
    with open('main.py') as f:
        for _ in f.readlines():
            if 'import' in _:
                exec(_)

def test_create():
    with open('create.py') as f:
        for _ in f.readlines():
            if 'import' in _:
                exec(_)
    
def test_debug():
    with open('debug.py') as f:
        for _ in f.readlines():
            if 'import' in _:
                exec(_)

        
def test_calibrate():
    with open('calibrate.py') as f:
        for _ in f.readlines():
            if 'import' in _:
                exec(_)
        