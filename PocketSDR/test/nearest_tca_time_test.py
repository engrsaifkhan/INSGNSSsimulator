import ast
from pathlib import Path


def load_function(path_str, func_name):
    path = Path(path_str)
    src = path.read_text(encoding='utf-8')
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(src, node)
    raise RuntimeError(f'Function {func_name} not found')


def run_test():
    ns = {
        'D2R': 3.141592653589793 / 180.0,
        'sol_log': [
            [10.0, [0.0, 0.0, 100.0], [8]],
            [20.0, [0.1, 0.0, 101.0], [9]],
            [30.0, [0.2, 0.0, 102.0], [10]],
        ],
        'tca_status': {},
        'tca_last_ref_pos': None,
        'tca_last_ref_source': '-',
        'tca_page': None,
        'tca_manual_ref_pos': None,
        'format_llh': lambda pos: 'ok',
        'sdr_rtk': type('SdrRtk', (), {'timediff': staticmethod(lambda a, b: abs(float(a) - float(b)))}),
    }

    func_src = load_function('python/pocket_sdr.py', 'tca_error_ref_mode')
    exec(func_src, ns)
    func_src = load_function('python/pocket_sdr.py', 'get_tca_error_reference')
    exec(func_src, ns)
    result, source = ns['get_tca_error_reference'](use_latest_pocket=True, tca_time=21.0)
    assert result == ns['sol_log'][1][1], (result, source)
    print('OK')


if __name__ == '__main__':
    run_test()
