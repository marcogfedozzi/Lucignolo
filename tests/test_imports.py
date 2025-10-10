# Minimal import smoke-test for lucignolo

import sys
from importlib import import_module
from typing import List

FAILS = []

def try_import(msg, fn):
		try:
				fn()
				print(f"OK: {msg}")
		except Exception as e:
				print(f"FAIL: {msg} -> {e!r}")
				FAILS.append((msg, e))

def check_module_content(mod, contents: List[str]):
		ok = True
		
		for c in contents:
				ok = ok and hasattr(mod, c)

				if not ok: 
					print("FAIL: ", mod, " - ", c)
					break
		
		if ok:
				print(f"OK ", mod)


# Top-level package
try_import('import lucignolo', lambda: import_module('lucignolo'))

# Subpackages and core
try_import('import lucignolo.core', lambda: import_module('lucignolo.core'))
try_import('from lucignolo.core import frames, utils, timers', lambda: import_module('lucignolo.core.frames') and\
						import_module('lucignolo.core.utils') and import_module('lucignolo.core.eef_point') and\
						import_module('lucignolo.core.timers'))

# Controllers
try_import('import lucignolo.controllers', lambda: import_module('lucignolo.controllers'))
try_import('from lucignolo.controllers import jnt_ctrl, invdyn_ctrl, multi_ctrl', lambda: \
					 import_module('lucignolo.controllers.base') and\
					 import_module('lucignolo.controllers.invdyn_ctrl') and\
					 import_module('lucignolo.controllers.jnt_ctrl') and\
					 import_module('lucignolo.controllers.multi_ctrl')
					)

mod = import_module('lucignolo.controllers')
check_module_content(mod, [
							'JointController',
							'ConstraintJointController',
							'IDController',
							'NoisyIDController',
							'MultiController',
							'SmoothMultiController']
					)

# Fields
try_import('import lucignolo.fields', lambda: import_module('lucignolo.fields'))


mod = import_module('lucignolo.fields')
check_module_content(mod, [
							'get_field',
							'XField',
							'FField',
							'VField']
					)
					
# Trajectory

mod = import_module('lucignolo.trajectory')
try_import('from lucignolo.trajectory import toys_path, mocap_traj_ctrl', lambda:\
					  import_module('lucignolo.trajectory.toys_path') and import_module('lucignolo.trajectory.mocap_traj_ctrl'))
check_module_content(mod, [
							'MocapControl',
							'AutonomousMocapControl',]
					)

print('\nSummary:')
if not FAILS:
		print('All imports succeeded')
		sys.exit(0)
else:
		print(f'{len(FAILS)} import(s) failed:')
		for name, err in FAILS:
				print('-', name, '->', err)
		sys.exit(1)
