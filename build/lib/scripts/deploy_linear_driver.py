import runpy
print(__file__)
runpy.run_module("src.apps.driver.driver_app", run_name="__main__", alter_sys=True)
