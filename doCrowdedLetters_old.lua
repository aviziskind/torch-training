doCrowdedLetters_old = function() -- crowdedLetterOpts, loadOpts, trainOpts)
--nstates = {6,16,120} --- ORIGINAL SET ---
--nstates = {9,24,180}	  --- x 1.5 ---
--nstates = {10,27,200}
--nstates = {12,32,240}	  --- x 2 ---
--nstates = {30,80,600}	  --- x 5 --- 
--nstates = {60, 160, 1200} -- x 10 --- physiological parameters --- 

    --allFontNames_ext = {'Braille', 'Sloan', 'Helvetica', 'Courier', 'Bookman', 'GeorgiaUpper', 'Yung', 'Kuenstler'}

    fontName = 'Bookman'
    fontSize = 12
    
    --useConvNet = expTitl == 'ConvNet'
    useConvNet = true
    stride = 1
    
    trainOpts.COST_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.TRAIN_ERR_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.MIN_EPOCHS = 100
    
    crowdedLetterOpts = {Nx = 44, Ny = 1, targetCentered = true, nDistractors = 1, exp_typ = 'Crowded'}

    
    letterOpts_str = getCrowdedLetterOptsStr(crowdedLetterOpts)


    --allNHiddenUnits = {3,4,5,6,7,8,9,10,20,50,100}
    if useConvNet then
        trainOpts.SWITCH_TO_LBFGS_AT_END = false
        trainOpts.REQUIRE_COST_MINIMUM = false
        loadOpts.totalUseFrac = 1
--        snr_train = {1,2,3,4}
        

        --allNStates = { {6, 50}, {6, 100},  {12, 50}, {12, 100},  {12,32,240},   {6, 16, 120}, {12,32,240}, {30,80,600}, {60, 160, 1200} }
        --allNStates = {  {6, 16, 120}, {9,24, 180}, {12,32,240} } -- {60, 160, 1200} }
        allNStates = {  {3, 8, 60} } -- {12, 32, 240} } -- }, {12,32,240}, {60, 160, 1200} } -- {60, 160, 1200} }
             
        allN_tnsr = tbltbl2Tensor(allNStates):double()

        
        --allNStates3 = {600}
        allN = allNStates
        
        --nstates = {30,80,600}	  --- x 5 --- 

        
    else
        trainOpts.SWITCH_TO_LBFGS_AT_END = false
        loadOpts.totalUseFrac = 1
        --snr_train = 2
        --snr_train = 'all';
        --snr_train = {3,4}
        
        allNHiddenUnits = {4,10,20,40,100,200}
        
        --allNHiddenUnits = {1,2, 3, 4, 5,10,25,50}

        allN = allNHiddenUnits
        allN_tnsr = torch.Tensor(allN)
    end
    --allNHiddenUnits = {10}
    nnUnits = #allN
        
        
    trainData, testData_contrasts = loadCrowdedLetters(fontName, fontSize, allSNRs, crowdedLetterOpts, loadOpts, true)
      
                
                               
    pct_correct_vs_snr = torch.Tensor(nnUnits, nSNRs):zero()
               
    for net_i = 1, nnUnits do
        net_i_copy = net_i
        print('Now doing network # ' .. net_i .. ' / ' .. nnUnits )
        if useConvNet then
            conv_nStates = allNStates[net_i]
            print('     ---------------- Conv network (' .. table.concat(conv_nStates, '_') .. ' ) ----------')
            networkOpts = {nInputs = trainData.nInputs, height = trainData.height, width = trainData.width,
                            nStates = conv_nStates, ConvNet = true, nClasses = trainData.nClasses, stride = stride}
            
        else
            nHiddenUnits = allNHiddenUnits[net_i]
            nHiddenUnits_str = hiddenLayer_str(nHiddenUnits)
            print('     ---------------- N HIDDEN UNITS = ' .. nHiddenUnits_str .. ' ----------------  ')
            
            print(string.format('Training network on Font = %s, # Hidden Units = %s', fontName, nHiddenUnits_str))
            networkOpts = {nInputs = trainData.nInputs, nHiddenUnits = nHiddenUnits, nClasses = trainData.nClasses}
        end
        
        model_struct = generateModel(networkOpts)
        expSubtitle = getExpSubtitle(fontName, nil, networkOpts, crowdedLetterOpts)
        print('Experiment subtitle : ' .. expSubtitle)

        model_struct = trainModel(model_struct, trainData, nil, trainOpts)   
        
        --errRate = testModel(model_struct, testData)
        
        for si,snr_test in ipairs(allSNRs) do
            io.write(string.format(' -- testing on snr = %.1f ...  ', snr_test))
           
            errRate_i = testModel_multipleLabels(model_struct, testData_contrasts[si])
            pct_correct_vs_snr[net_i][si] = 100 - errRate_i
            io.write(string.format('pct correct = %.1f %%\n', pct_correct_vs_snr[net_i][si]))
        end
    end
                --visualizeHiddenUnits(model)
                
    print('=================== Summary for Font = ' .. fontName)
    print('all nStates  : ')
    print(allN_tnsr)
    print('% correct ')
    print(pct_correct_vs_snr)
    
    if useConvNet then
        var_list = {allNStates = allN_tnsr:double(), pct_correct_vs_snr = pct_correct_vs_snr:double()}
    else
        var_list = {allNHiddenUnits = allN_tnsr:double(), pct_correct_vs_snr = pct_correct_vs_snr:double()}
    end
    
    save_filename = expTitl .. '_' .. fontName .. '__' .. letterOpts_str  .. '.mat'
    print('Saving results to ' .. save_filename)
    mattorch.save(results_dir .. save_filename, var_list)
                
               
  
end