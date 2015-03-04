print("Initializing Torch Session ... ")

-- Navigate to letters folder
require 'lfs'
--if not string.find( paths.cwd(),  'letters') then
    print('-Moving to letters/torch folder... ')
    lfs.chdir(paths.home .. '/Code/nyu/letters/torch/')
--end

print('-Loading common scripts ... ')
dofile (paths.home .. '/Code/myscripts/torch/load_all_torch_scripts.lua')

hostname = os.getenv('hostname') 
onLaptop = (hostname == 'XPS')

-- creating session file
local sessionFolder =  paths.home .. '/Code/~locks/session/'
createFolder(sessionFolder)
session_file_name = sessionFolder .. 'torch_session_' .. os.getenv('host') .. '_' .. os.getenv('bashpid')
createSessionFile = function()      f = assert( io.open(session_file_name, 'w') )   end

session_file_alreadyExists = paths.filep(session_file_name)

if session_file_alreadyExists then
    session_file_existed_str = 'already exists'
else
    session_file_existed_str = 'just created'
    createSessionFile()
end

print(string.format('-"reload-session" file :\n  %s (%s)', paths.basename(session_file_name), session_file_existed_str ))

        
 
 
local runGPUtests = false

if runGPUtests then
    local doLoadCutorch = false
    local doTestGetParameters = false
    local doResetDevice = false
    
    print('-Testing GPU ...')
    if doLoadCutorch then
        print('  (1) Loading cutorch library ... ');
        require 'cutorch'    
        print('  --Done!')
    end
    
    if doTestGetParameters then 
        print('  (2) Testing getParameters ... ');
        require 'cunn'
        local m = nn.Sequential()
        m:add(nn.Linear(2,5))
        m:cuda()
        local x,dx = m:getParameters()
        print('  --Done!')
    end
    
    if doResetDevice then
        print('  (3) Resetting Device')
        
        local GPU_ID = os.getenv('GPU_ID')
        if GPU_ID then
            local device_id = GPU_ID + 1
            print(string.format('   Detected GPU_ID=%d ==> Selecting GPU #%d', GPU_ID, device_id))
            cutorch.setDevice(device_id)
            print('       done!')
            assert( cutorch.getDevice(device_id) == device_id)
            print('   Resetting device')
            cutorch.deviceReset()
            print('       done!')
        end
    end
end


go = function()
    if not paths.filep(session_file_name) then
        createSessionFile()
    end
    
    dofile 'main.lua'
    
    if paths.filep(session_file_name) then
        os.execute('rm ' .. session_file_name )
    end
end

if session_file_alreadyExists or os.getenv('GO') then
    print('Detected session file at startup. Resuming main.lua ...')
    go()
end



--[[


require 'nn'

lin = nn.Linear(4096, 30)
input = torch.FloatTensor(4096)
output = lin:forward(input)




require 'nn'
ninputs = 406
noutputs = 30
model = nn.Linear(ninputs, noutputs):float(); input = torch.randn(ninputs):float()  output = model:forward(input)



t7> =string.find('ls: cannot access /home/ziskind/Code/torch/locks/Converting_Bookman-k23_x-34-15-266_T4t18-[231x231]_SNR20_OF__1let.lock*: No such file or directory', lfn)
                                                                   Converting_Bookman-k23_x-34-15-266_T4t18-[231x231]_SNR20_OF__1let.lock	

--]]
