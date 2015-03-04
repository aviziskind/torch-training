dofile 'letters_scripts.lua'
dofile 'load_data.lua'

convertAllOverFeatFiles = function(matlab_dir, convertOpts)
    
    local redo = false
    local redoIfOld = false
    local redoIfOld_date = 1409599369.762
        
    require 'lfs'        

    if not matlab_dir then
        nDoneTotal = 0
        nDoneMax = 1
        lock.createLock(ID)
        
        local tbl_allContrasts = {127, 64, 32, 16, 2, 1}
        local tbl_allOffsets = {127, 0, 64}
        local tbl_allNetworkIds = {0}
        local layerIds = {19, 17, 16}

        local allConvertOpts = expandOptionsToList( {tbl_OF_offset = tbl_allOffsets, 
                                                     tbl_OF_contrast=tbl_allContrasts, 
                                                     tbl_networkId = tbl_allNetworkIds,
                                                     layerIds = layerIds,
                                                     doOverFeat = true}, {'OF_offset', 'OF_contrast'} ) 
                                             
        
        print(allConvertOpts)
        
        if not matlab_datasets_dir then
            home_dir = paths.home
            torch_dir = home_dir ..'/Code/torch/'
            torchLetters_dir = torch_dir .. 'letters/'    

            hostname = os.getenv('hostname') 
            onLaptop = (hostname == 'XPS')


            print('Running load_data & letters_scripts scripts')
            dofile (torchLetters_dir .. 'avi_scripts.lua')
            dofile (torchLetters_dir .. 'load_data.lua')
            dofile (torchLetters_dir .. 'letters_scripts.lua')
            
            if onLaptop then
                
                matlabLetters_dir = home_dir .. '/Code/MATLAB/nyu/letters/'
                matlab_datasets_dir = matlabLetters_dir .. 'datasets/'
                
                overfeat_weights_dir = '/usr/local/overfeat/data/default/'

                --torch_datasets_dir = torchLetters_dir .. 'datasets/'  
                
            else
                nyu_data_dir = '/misc/vlgscratch2/LecunGroup/ziskind/lettersData/'
                matlab_datasets_dir = nyu_data_dir .. 'MATLAB/datasets/'              
                
                overfeat_weights_dir = nyu_data_dir .. 'torch/overfeat/data/default/'
                
                --nyu_data_dir = '/home/ziskind/lettersData/' -- this is a symlink to scratch location
                --torch_datasets_dir = nyu_data_dir .. 'torch/datasets/'  
                
            end
        end       
        
        for i,opt in ipairs(allConvertOpts) do
            io.write(string.format('\n\n ====================== %d / %d : %s ============ \n', i, #allConvertOpts, getOverFeatStr(opt)) )
            
            local matlab_dir1 = matlab_datasets_dir .. 'NoisyLettersOverFeat/'
            local matlab_dir2 = matlab_datasets_dir .. 'CrowdedLetters/'
            convertAllOverFeatFiles(matlab_dir1, opt)
            convertAllOverFeatFiles(matlab_dir2, opt)
        end        
        return
        --torch_dir = torch_datasets_dir
    end
    
    local convertOpts_allLayers = {}
    local overfeat_file_exts = {}
    for i = 1,#convertOpts.layerIds do
        convertOpts_allLayers[i] = table.copy(convertOpts)
        convertOpts_allLayers[i].layerId = convertOpts.layerIds[i]
        overfeat_file_exts[i] = getOverFeatStr(convertOpts_allLayers[i])
    end
--    local overfeat_file_ext = getOverFeatStr(convertOpts)
        
    local overfeatMatlabImageFiles = dir(matlab_dir .. '*_OFim*.mat')    
        
    local nFiles = #overfeatMatlabImageFiles
    if nFiles > 0 then
        --print(overfeatMatlabFiles)
        ---[[
        for i,matImFileBase in ipairs(overfeatMatlabImageFiles) do
            local matImFile = matlab_dir .. matImFileBase
            
            io.write(string.format(' (%d/%d) %s => ', i, nFiles, matImFileBase))
            
            local matFeatFilesBase, matFeatFiles, matFeatFileNeedToDo = {}, {}, {}
            for i,overfeat_file_ext in ipairs(overfeat_file_exts) do
                matFeatFilesBase[i] = string.gsub(matImFileBase, '_OFim', overfeat_file_ext)
                matFeatFiles[i] = matlab_dir .. matFeatFilesBase[i]
                matFeatFileNeedToDo[i] = not paths.filep(  matFeatFiles[i] )  or fileOlderThan(matFeatFiles[i], matImFile) or 
                    (redoIfOld and fileOlderThan(matFeatFiles[i], redoIfOld_date))
                    
                    
            end
            
            
            
            
            
            if table.any(matFeatFileNeedToDo) or redo  then
                io.write('\n')
                for i,overfeat_file_ext in ipairs(overfeat_file_exts) do    
                    io.write(string.format('    [%d] %s %s\n', i, matFeatFilesBase[i], iff(matFeatFileNeedToDo[i], '[Need to do]', '[Done]' )) )
                end

            
                io.write('  Converting ... \n')
                local lock_name = 'Converting_' .. matImFileBase
                local gotLock, otherProcessID = lock.createLock(lock_name)
                if gotLock then
                    createOverFeatFeaturesFile(matImFile, matFeatFiles, convertOpts)
                    lock.removeLock(lock_name)
                    
                    collectgarbage('collect')  -- train/test data can be very big -- after done with one font, 
                    collectgarbage('collect')  -- clear memory for next font data

                    
                    nDoneTotal = nDoneTotal + 1
                    if nDoneTotal >= nDoneMax then
                        print(string.format('Reached max # of targets (%d). Exiting...', nDoneTotal))
                        sys.sleep(5);
                        lock.removeLock(ID)
                        exit()
                    end
                    --convertMatToTorch(matFeatFile, torchFile, convertOpts)
                else
                    io.write(string.format('Another process [%s] has lock on %s\n', otherProcessID, lock_name))
                end
                    
            else
                
                io.write('  [Completed] \n')
            end
                
        end
        --]]
        
    end
    
    
    -- recurse
    local subs_matlab = subfolders(matlab_dir)
    --local subs_torch = subfolders(torch_dir)
    
    --local subs_inCommon = table.intersect(subs_torch, subs_matlab)
    
    for i,subfolder in ipairs(subs_matlab) do
        print('Searching in : ', matlab_dir .. subfolder)
        --convertAllOverFeatFiles(matlab_dir .. subfolder .. '/', torch_dir .. subfolder .. '/') 
        convertAllOverFeatFiles(matlab_dir .. subfolder .. '/', convertOpts) 
    end
    
    
    
    
end