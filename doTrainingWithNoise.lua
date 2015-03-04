doTrainingWithNoise = function() --- allFontNames, allSNRs, loadOpts, noisyLetterOpts, trainOpts)    

    local fontNames_use = {'Bookman'}
    --fontName = 'Braille'
    
    --allsnrs = { 0, 1, 2, 3, 4 }
    local allSNRs_train = { {0}, {1}, {2}, {3}, {4}, 
                        --{0, 1}, {1, 2}, 
                      {2, 3}, {3, 4}, 
                      {0, 1, 2},  {1, 2, 3}, {2, 3, 4}, {1, 2, 3, 4},
                      {0, 1, 2, 3, 4}, 
                      {0, 1, 1.5, 2, 2.5, 3, 4}, } 
    --nSNRsets = #allSNRs_train
    --nSNRs = #allSNRs
    --nTrials = 1
    
    
    noisyLetterOpts = {Nori = 1, ori_range = 0,
                        Nx = 1,   x_range = 0,
                        Ny = 1,   y_range = 0, exp_typ = 'Noisy'}
    
    
    local allSNRs_test = allSNRs
    --fontNames = {'Kuenstler', 'Yung'}
    --fontNames = {'Bookman', 'GeorgiaUpper'}
    --fontNames = {'Bookman'}
    --local useConvNet = expTitl == 'ConvNet'
    
    trainOpts.COST_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.TRAIN_ERR_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.MIN_EPOCHS = 20
    
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    trainOpts.REQUIRE_COST_MINIMUM = false
    loadOpts.totalUseFrac = 1

    local allSizeStyles = {'sml'}

   
    
    --allNHiddenUnits = {3,4,5,6,7,8,9,10,20,50,100}
    
    local networkType = 'MLP'
    local allNetworks
    
    if networkType == 'ConvNet' then

        local poolStrides = 2
        local doPooling = true
        local filtSizes = {5,4}
        local snr_train, noisyLetterOpts

        local allNStates = { {6, 16, 120}  }
        local allConvNetOptions = { netType = 'ConvNet', 
                              poolStrides = poolStrides, 
                              doPooling = doPooling, 
                              filtSizes = filtSizes,
                              tbl_nStates = allNStates
                            }
               
        allNetworks = expandOptionsToList(allConvNetOptions)
        
        
    elseif networkType == 'MLP' then
        
        local allNHiddenUnits = { {10, 10} }
            
        allMLPoptions = { netType = 'MLP',
                          tbl_nHiddenUnits = allNHiddenUnits }
                    
        allNetworks = expandOptionsToList(allMLPoptions)
                
    end
    
    print(allNetworks)
        
    
    local all_OriXY =  {  {1,0,  1,0,  1,0  }
                        --[[{1,0,  2,3,  1,0  }, 
                        {1,0,  3,6,  1,0  }, 
                        {1,0,  7,6,  1,0  }, 
                        {1,0,  7,6,  7,6  }, 
                        {1,0,  10,9, 10,9 }, 
                        {5,8,  1,0,  1,0  }, 
                        {5,8,  5,4,  5,4  }, 
                        {5,8,  10,9, 10,9 } 
                       --]]
                       }             
    
    --doTrainingBatch(allFontNames, allSNRs, allNetworks, allOriXY, allSnrTrain, allSizeStyles)
    print(allSizeStyles)
    doTrainingBatch(fontNames_use, allSNRs_train, allSNRs_test, allNetworks, all_OriXY, allSizeStyles)
    --doTrainingBatch(fontNames, allSNRS_train, allSNRs_test, allNetworks, all_OriXY, allSizeStyles)  
  
  
end



--[[

doTrainingWithNoise = function() --- allFontNames, allSNRs, loadOpts, noisyLetterOpts, trainOpts)    

    fontName = 'Bookman'
    --fontName = 'Braille'
    
    --allsnrs = { 0, 1, 2, 3, 4 }
    allSNRs_train = { {0}, {1}, {2}, {3}, {4}, 
                    --{0, 1}, {1, 2}, 
                    {2, 3}, {3, 4}, 
                    {0, 1, 2},  {1, 2, 3}, {2, 3, 4}, {1, 2, 3, 4},
                    {0, 1, 2, 3, 4}, 
                    {0, 1, 1.5, 2, 2.5, 3, 4}, } 
    nSNRsets = #allSNRs_train
    nSNRs = #allSNRs
    nTrials = 2
    
    useConvNet = 1
    
    
    noisyLetterOpts = {Nori = 1, ori_range = 0,
                        Nx = 1,   x_range = 0,
                        Ny = 1,   y_range = 0, exp_typ = 'Noisy'}
    
    
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    
---[[

    local allNStates, allNHiddenUnits, nNetworks

    if useConvNet then
        trainOpts.SWITCH_TO_LBFGS_AT_END = false
        trainOpts.REQUIRE_COST_MINIMUM = false
        loadOpts.totalUseFrac = 1

        --allNStates = { {6, 50}, {6, 100},  {12, 50}, {12, 100},  {12,32,240},   {6, 16, 120}, {12,32,240}, {30,80,600}, {60, 160, 1200} }
        --allNStates = {  {6, 16, 120}, {9,24, 180}, {12,32,240} } -- {60, 160, 1200} }
        --allNStates = {  {3, 8, 60}, {6, 16, 120}, {12,32,240} } -- {60, 160, 1200} }
        allNStates = {   {6, 16, 120} } -- {60, 160, 1200} }
        allConvNetOptions = {  {
                     
        nNetworks = #allNStates
    else
        trainOpts.SWITCH_TO_LBFGS_AT_END = false
        loadOpts.totalUseFrac = 1
        --snr_train = 2
        --snr_train = 'all';
        --snr_train = {1,2,3}
        
        allNHiddenUnits = { 10, {10,10}}
        nNetworks = #allNHiddenUnits
    end
    --allNHiddenUnits = {10} --, 25, 100}
    nnU = #allNHiddenUnits
    
    pctCorr_v_snr = torch.Tensor(nnU, nSNRsets, nSNRs, nTrials)
    trainData = {}
    testData = {}
    testData_noisy = {}
    
    noisyLetterOpts_str = getNoisyLetterOptsStr(noisyLetterOpts)
    
    createFolder(results_dir)
  
    
    for s_i,snr_to_do in ipairs(allSNRs) do
        io.write(string.format('Loading SNR = %.1f : ', snr_to_do))
        trainData_tmp, testData_noisy[s_i] = loadNoisyLetters(fontName, snr_to_do, noisyLetterOpts, loadOpts)
    end
    
    --errRates_snr_v_snr = torch.Tensor(nSNRsets, nSNRs, nTrials):zero()
    
    for trial_i = 1,nTrials do
    print('           Trial # ' .. trial_i)

        for net_i, nstates in ipairs(allN) do

            print('Now doing network # ' .. net_i .. ' / ' .. #allN )
            if useConvNet then
                conv_nStates = allNStates[net_i]
                print('     ---------------- Conv network (' .. table.concat(conv_nStates, '_') .. ' ) ----------')
                networkOpts = {nInputs = trainData_tmp.nInputs, height = trainData_tmp.height, width = trainData_tmp.width,
                                nStates = conv_nStates, ConvNet = true, nClasses = trainData_tmp.nClasses}
                
            else
                nHiddenUnits = allNHiddenUnits[net_i]
                nHiddenUnits_str = hiddenLayer_str(nHiddenUnits)
                print('     ---------------- N HIDDEN UNITS = ' .. nHiddenUnits_str .. ' ----------------  ')
                
                print(string.format('Training network on Font = %s, SNR = %s, # Hidden Units = %s', fontName, tostring(snr_train), nHiddenUnits_str))
                networkOpts = {nInputs = trainData.nInputs, nHiddenUnits = nHiddenUnits, nClasses = trainData.nClasses}
            end
            

        --nUnits_tnsr = torch.Tensor(allHiddenUnits)    
           
            for s_i,snr_to_do in ipairs(allSNRsets) do
                if type(snr_to_do) == 'table' then
                    snr_to_do_str = table.concat(snr_to_do, ', ')
                else
                    snr_to_do_str = tostring(snr_to_do)
                end
                print('     Training on SNR set = ' .. snr_to_do_str)
                
                trainData, testData = loadNoisyLetters(fontName, snr_to_do, noisyLetterOpts, loadOpts)
                

                model_struct = generateModel(networkOpts)
                expSubtitle = getExpSubtitle(fontName, snr_to_do, networkOpts, noisyLetterOpts, trial_i)

                model_struct = trainModel(model_struct, trainData, testData, trainOpts)
                
               
                for s_j,snr_test in ipairs(allSNRs) do
                    io.write(string.format(' -- testing on snr = %.1f ...  ', snr_test))
                   
                    errRate_i = testModel(model_struct, testData_noisy[s_j])
                    pctCorr_v_snr[net_i][s_i][s_j][trial_i] = 100 - errRate_i
                    io.write(string.format('pct correct = %.1f %%\n', pctCorr_v_snr[net_i][s_i][s_j][trial_i]))
                end
                
            end    
            
            --errRate = testModel(model, testData, verbose)
            --errRates_v_snr[s_i] = errRate
            
        end
         
        var_list = {allNHiddenUnits = torch.Tensor(allNHiddenUnits):double(),
            pctCorr_v_snr = pctCorr_v_snr[{{}, {}, {}, {1, trial_i} }]:double(),
            allSNRsets = tbltbl2Tensor(allSNRsets, -1):double()}
        save_filename = expTitl .. network_type .. '_' .. fontName .. '__' .. noisyLetterOpts_str  .. '.mat'

        print(string.format('Saving results from first %d trials to %s', trial_i, save_filename))
        mattorch.save(results_dir .. save_filename, var_list) 
         
         
    end

  


            --allNHiddenUnits_tnsr = torch.Tensor(allNHiddenUnits)
   
end

--]]