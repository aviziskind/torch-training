local cmd = torch.CmdLine()
cmd:option('-THREAD_ID', -1, 'thread id of this torch instance')
cmd:option('-N_THREADS_TOT', -1, 'total number of threads run')
--cmd:option('-N_CPUs', 8, 'max iterations')
local cmdLineOpts = cmd:parse(arg or {})


print(cmdLineOpts)
N_THREADS_TOT = cmdLineOpts.N_THREADS_TOT
THREAD_ID = cmdLineOpts.THREAD_ID

--sys.sleep(THREAD_ID) -- help prevent conflicts

dofile 'avi_scripts.lua'
--------- insert code you want to run here --------------


dofile 'main.lua'


--dofile 'mainHelper.lua'

--lock.testLock()

--dofile 'avi_scripts.lua'
--progressBar.test()

---[[

--]]



---------------------------------------------------------

--print('[Successfully completed torch session; Press <Return> to exit]')