torch_dir = '/home/avi/Code/torch/letters/'
stop_file = torch_dir .. 'stop_running'
stop_file_tmp = torch_dir .. 'stop_runnin'
    

go = function()
    if paths.filep(stop_file) then
        os.rename(stop_file, stop_file_tmp)
    end
        
    j = 0
    while true do
     
        for i = 1,1000000 do
            j = j + 10
        end
     
        io.write('*');
     
        if paths.filep(stop_file) then
            error('Stop.')
        else
            print('[File not present]')
        end
     
        
    end
     
    
end

