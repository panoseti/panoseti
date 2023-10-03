#! /usr/bin/env python3

import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd(r'all_sky_matlab', nargout=0)
webcrep = "SC_imgs/"
dateproc = "20230801"
eng.panoplots(webcrep, 1.0, dateproc)
eng.quit()