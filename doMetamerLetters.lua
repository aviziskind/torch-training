doMetamerLetters = function() -- (allFontNames, allSNRs, loadOpts, noisyLetterOpts, trainOpts)
--nstates = {6,16,120} --- ORIGINAL SET ---
--nstates = {9,24,180}	  --- x 1.5 ---
--nstates = {10,27,200}
--nstates = {12,32,240}	  --- x 2 ---
--nstates = {30,80,600}	  --- x 5 --- 
--nstates = {60, 160, 1200} -- x 10 --- physiological parameters --- 

    --allFontNames_ext = {'Braille', 'Sloan', 'Helvetica', 'Courier', 'Bookman', 'GeorgiaUpper', 'Yung', 'Kuenstler'}

    local fontNames_use = 'BookmanUpper'
    
    local netType = 'ConvNet'
--    local netType = 'MLP'
    
    
    trainOpts.COST_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.TRAIN_ERR_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.MIN_EPOCHS = 50
    
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    trainOpts.REQUIRE_COST_MINIMUM = false
    loadOpts.totalUseFrac = 1

    -- size styles to do
    
    --allNHiddenUnits = {3,4,5,6,7,8,9,10,20,50,100}
        
    local allNetworks
    local allNStates, allNetNames 
    if netType == 'ConvNet' then
        -- ConvNet options

        
        
        --        snr_train = {1,2,3,4}
        snr_train = {1,2,3}

        --snr_train = {1,2,3,4}
        --snr_train = {0,1,2,3,4}


        --allNStates = { {6, 50}, {6, 100},  {12, 50}, {12, 100} }
        --allNStates = { {6, 100}  }
        --allNStates = {  {6, 16, 120}, {9,24, 180}, {12,32,240} } -- {60, 160, 1200} }
        --allNStates = {  {6, 16, 120}, {12,32,240}, {60, 160, 1200} } -- {60, 160, 1200} }
        
        --allNStates = { {3, 8, 60}, {6, 16, 120}, {6, 16, 30}, {12,32,240}, } -- {60, 160, 1200} }
        --allNStates = {  {6, 16, 120}, } --  {12, 32, 120}, {24, 64, 120},  {48, 128, 120} } 
        --allNStates = {  {6, 16, 15},  {6, 16, 30}, {6, 16, 60},  {6, 16, 120}, {6, 16, 240}, {6, 16, 480}, {6, 16, 960} } -- {60, 160, 1200} }
        
        --allNStates = {  {6, 16, 120},  {6, 32, 120}, {6, 64, 120},  {6, 128, 120}, {6, 16, 960}  } -- {60, 160, 1200} }
        --allNStates = {  {3, 16, 120}, {6, 16, 120},  {12, 16, 120}, {24, 16, 120}, {48, 16, 120},   } -- {60, 160, 1200} }
        --allNStates = {  {3, 8, 60}, {6, 16, 120}, {6, 16, 30}, {6,16, 15}, {6,16,8}, {12,32,240},  {3,8,10}, {3,8,5}, {3,8,3}, {3,8,30}, {6,8,10}, {12,8,10} }
        --allNStates = {  {6, 16, 120}, {}
            
        --allNStates = { {6, 16, 960} ,
          --              {6, 4, 120},  {6, 8, 120},  {6, 16, 120},  {6, 32, 120}, {6, 64, 120},  {6, 128, 120}, {6, 256, 120},  -- {60, 160, 1200} }
            --             {3, 16, 120}, {6, 16, 120},  {12, 16, 120}, {24, 16, 120}, {48, 16, 120}, {96, 16, 120}, {192, 16, 120}  } -- {60, 160, 1200} }
            
        --local allNStates = { {6, 15}, {6, 30}, {6,60}, {6,120}, {6, 240}, {12, 15}, {12, 30}, {12,60}, {12,120}, {12, 240} }
        
        local allNStates = { {6, 15} }

        local filtSizes = {5, 4}
        local allFiltSizes = {3, 5, 10}
        
        local doPooling = true
        local allDoPooling = {false, true}

        local poolSize = {4, 2}
        
        --local filtSizes = {34}
        
        local poolType = 2
        local allPoolTypes = {1, 2, 'MAX'}
        
        local poolStride = 2 --'auto'
        local allPoolStrides = {'auto', 2,4,6}
            
        local allPoolSizes = {0, 8,16,24}

        local allConvNetOptions = { netType = 'ConvNet', 
                                    tbl_nStates = allNStates,
                                    filtSizes = filtSizes,
                                    --tbl_filtSizes = allFiltSizes,
                                    
                                    doPooling = doPooling, 
                                    --tbl_doPooling = allDoPooling,
                                    
                                    --poolSizes = poolSize,
                                    tbl_poolSizes = allPoolSizes,
                                    
                                    poolStrides = poolStride, 
                                    --tbl_poolStrides = allPoolStrides,
                                    
                                    poolType = poolType,
                                    --tbl_poolType = allPoolTypes
                                  }
               
               
        local allNetworks_orig = expandOptionsToList(allConvNetOptions)
        local allNetworks_fixed = fixNetworkParams(allNetworks_orig)
        
        allNetworks, allNetNames = uniqueNetworks(allNetworks_fixed)
                
        print(allNetNames)
        
        
        AllNetworks = allNetworks
 
        --allNetworks_tnsr = tbltbl2Tensor(allNStates):double()

        
        --allNStates3 = {600}
        --allNetworks = allNStates
        
        --nstates = {30,80,600}	  --- x 5 --- 
    elseif netType == 'MLP' then
        
        local allNHiddenUnits = { {6}, {12}, {24}, {48}, {96},   {6, 16}, {6, 32}, {6, 64},   {12, 16}, {12, 32}, {12, 64} }
            
  --          {6, 15}, {6, 30}, {6,60}, {6,120}, {6, 240}, {12, 15}, {12, 30}, {12,60}, {12,120}, {12, 240} }
        local allMLPoptions = { netType = 'MLP', 
                                tbl_nHiddenUnits = allNHiddenUnits }
                    
        allNetworks = expandOptionsToList(allMLPoptions)
        
--        allNHiddenUnits = {4,10,20,40,100,200}
        --allNHiddenUnits = {1,2, 3, 4, 5,10,25,50}
        
    end
    
    
    
    local fontName = 'BookmanUpper'
    local metamerLetterOpts = {size = 64, niter = 200, stimType = 'Metamer'}
        
    --doMetamerTrainingBatch(fontNames, allNetworks, metamerLetterOpts)
        --function(fontNames, allSNRS_train, allSNRs_test, allNetworks, all_OriXY, allSizeStyles)
    
  --]]  
  
  
    local redoResultsFiles = false
       
    
    local inputStats = getMetamerLettersStats(fontName, metamerLetterOpts)
    
    local trainData = nil
    local testData = nil
    
    createFolder(results_dir)
       
    for net_i, networkOpts_i in ipairs(allNetworks) do
        
        local networkStr = getNetworkStr(networkOpts_i)
        io.write(string.format('  ---------  Network # %d / %d : %s :   \n', net_i, #allNetworks, networkStr))
        
        --networkOpts_i_copy = networkOpts_i
       
        expSubtitle = getExpSubtitle(fontName, nil, networkOpts_i, metamerLetterOpts)
        
        expFullTitle = stimType .. '_' .. expSubtitle
        local save_filename_base = expFullTitle .. '.mat'
        local save_filename = results_dir .. save_filename_base

        if paths.filep(save_filename) and not redoResultsFiles then
            io.write(string.format('      =>Already completed : %s\n', save_filename_base))
                
        elseif not paths.filep(save_filename) or redoResultsFiles then
            
            local gotLock = lock.createLock(expFullTitle)
            if not gotLock then 
                io.write(string.format(' [Another process is already doing %s]\n', expSubtitle))
            
            else
                                    
                print('Experiment subtitle : ' .. expSubtitle)

                if not trainData then  -- load training/testing data for this font if don't have already
                    print('Loading training / test data ... ')
                    
                    trainData, testData = loadMetamerLetters(fontName, metamerLetterOpts, loadOpts)
                    TrainData = trainData
                    InputStats = inputStats
                    assert(trainData.nInputs == inputStats.nInputs)
                    assert(trainData.nClasses == inputStats.nClasses)
                    assert(trainData.height == inputStats.height)
                    assert(trainData.width == inputStats.width)
                                                
                end

                local model_struct = generateModel(inputStats, networkOpts_i)
                model_struct = trainModel(model_struct, trainData, testData, trainOpts)   
                
                io.write(string.format(' -- testing : ...  '))
                local errRate = testModel(model_struct, testData)
                local pct_correct = 100 - errRate 
                io.write(string.format('pct correct = %.1f %%\n', pct_correct))
                
                                                                        
                local var_list = {pct_correct = torch.DoubleTensor({pct_correct}) }
                
                --print(var_list)
                print('Saving results to ' .. paths.basename(results_dir) .. '/' .. save_filename)
                mattorch.save(save_filename, var_list)
                
                lock.removeLock(expFullTitle)
                                        
            end
        
        end  -- if don't have results file
        
    end -- loop over networks  (nNetworks)
    
    collectgarbage('collect')  -- train/test data can be very big -- after done with one font, 
    collectgarbage('collect')  -- clear memory for next font data
         
  
  
end


fixNetworkParams = function(allNetworks)
        
    for i,net in ipairs(allNetworks) do
        
        if (net.poolStrides ~= 'auto') and (net.poolStrides > net.poolSizes) then
            allNetworks[i].poolStrides = allNetworks[i].poolSizes
        end
        if (net.poolSizes == 0) then
            allNetworks[i].poolStrides = 0
            allNetworks[i].doPooling = false
        end
    end
    
    return allNetworks
end
        
        
uniqueNetworks = function(allNetworks)

    local tbl_networkNames = {}
    local tbl_networkNames_nice = {}
    for i,net in ipairs(allNetworks) do
        local net_str, net_str_nice = getNetworkStr(net);
        
        table.insert(tbl_networkNames, net_str)
        table.insert(tbl_networkNames_nice, net_str_nice)
    end

    local uniqueNames, idx_unique = tbl_unique(tbl_networkNames)
    
    local uniqueNetworks = table.subsref(allNetworks, idx_unique)
    local uniqueNetworkNames = table.subsref(tbl_networkNames_nice, idx_unique)
    return uniqueNetworks, uniqueNetworkNames

end
        
       
    
    
--[[

if THREAD_ID and (math.mod(fi-1, N_THREADS_TOT)+1 ~= THREAD_ID) then
                print(string.format(' ** Thread %d, skipping font # %d ** \n', THREAD_ID, fi))
                
            else
                if THREAD_ID then
                    print(string.format(' ** Thread %d, doing font # %d ** \n', THREAD_ID, fi))
                end
            
            
            --]]