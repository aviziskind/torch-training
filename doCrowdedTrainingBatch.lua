doCrowdedTrainingBatch = function(allNetworks, all_LetterOpts, allSNRs_test, nTrials)
--doCrowdedTrainingBatch(fontNames_use, allSNRs_test, allDNRs_test, allNetworks, all_LetterOpts, nTrials)    
    
    local redoResultsFiles = false
    local redoResultsFilesIfOld = true
    local redoResultsIfBefore = 1399992758 -- 1390594996   -- =os.time()
           
    --local nFonts = #fontNames
    local nSNRs = #allSNRs_test
    local nLetOpts = #all_LetterOpts
    local nNetworks = #allNetworks
    
    local saveNetworkToMatfile = true --true and (stimType ~= 'NoisyLettersTextureStats') -- and (NetType == 'ConvNet')

    local nToDo_total = nLetOpts * nNetworks * nTrials   
    local glob_idx = 1
    local curFont
    local doMultiLetters = true
       
       
    if not doMultiLetters then
        print('[ONLY DOING SINGLE LETTER TRAINING]');
    end
        
    for trial_i = 1,nTrials do
            
        for opt_i, crowdedLetterOpts in ipairs(all_LetterOpts) do        
            
            local fontNamesSet = crowdedLetterOpts.fontName
            assert(type(fontNamesSet) == 'table')                
            local fontNamesList = getFontList(fontNamesSet)
                            
            local fontName1 = fontNamesList[1]

            
            local sizeStyle = crowdedLetterOpts.sizeStyle
            
            local fontWidth = getFontAttrib(fontName1, sizeStyle, 'width')
            
            local SNR_train = crowdedLetterOpts.SNR_train
            
            local snr_train_str =  string.format('SNR-train = %s. ', toOrderedList(SNR_train))
            local crowdedLetterOpt_str = getCrowdedLetterOptsStr(crowdedLetterOpts)
                            
            local crowdedLetterOpts_1let = table.copy(crowdedLetterOpts)
            --local crowdedLetterOpts_2let = table.copy(crowdedLetterOpts)
            
            crowdedLetterOpts_1let.nLetters = 1
            local inputStats = getDatafileStats(crowdedLetterOpts_1let)
            
            local train_1let_trData = nil
            local train_1let_tsData = nil
            
            local test_1let_tsData = nil
            local test_2let_tsData = nil
            
            io.write(string.format('============== Stim: %s (%d/%d). FONT = %s. Size = %s ============  \n', 
                    crowdedLetterOpt_str, opt_i, #all_LetterOpts, fontName1, tostring(sizeStyle) ))
           
            
            for net_i, networkOpts_i in ipairs(allNetworks) do
               
                local networkStr = getNetworkStr(networkOpts_i)
                io.write(string.format('  --------- [%d / %d (%.1f%%)]   Network # %d / %d : %s :   \n', 
                        glob_idx,nToDo_total,glob_idx/nToDo_total*100,   net_i, #allNetworks, networkStr))                           
                    glob_idx = glob_idx + 1
                                    
                expSubtitle = getExpSubtitle(crowdedLetterOpts, networkOpts_i, trial_i)
                                
                expFullTitle = stimType .. '_' .. expSubtitle
                local results_dir_main = paths.dirname(results_dir) .. '/'
                local preferred_results_subdir = basename(results_dir) .. '/'
                        
                local results_filename_base = expFullTitle .. '.mat'
                local resultsFileExists, results_filename = fileExistsInPreferredSubdir(results_dir_main, preferred_results_subdir, results_filename_base)
                
                 
                --local results_filename = results_dir .. results_filename_base
                --local resultsFileExists = paths.filep(results_filename)
                
                local network_matfile = training_dir .. expSubtitle .. '.mat'
                local networkFileExists = paths.filep(network_matfile)
                
                local resultsFileTooOld = resultsFileExists and redoResultsFilesIfOld and 
                        (lfs.attributes(results_filename).modification < redoResultsIfBefore)
                        
                local networkFileTooOld = networkFileExists and redoResultsFilesIfOld and 
                        (lfs.attributes(network_matfile).modification < redoResultsIfBefore)
                                                            
                    
                local allMultiLetOpt = crowdedLetterOpts.allMultiLetOpt
                local haveAllFiles_2let = true
                
                local allMultiLetterResultsFileNames = {}
                if doMultiLetters then
                    for i,optMultiLet in ipairs(allMultiLetOpt) do
                        local expSubtitle_2let = getExpSubtitle(optMultiLet, networkOpts_i, trial_i)
                        local results_filename_base_2let = stimType .. '_' .. expSubtitle_2let .. '.mat'
                        
                        local resultsFileExists_2let, results_filename_2let = fileExistsInPreferredSubdir(results_dir_main, preferred_results_subdir, results_filename_base_2let)
                        
                        allMultiLetterResultsFileNames[i] = results_filename_2let
                        --local results_filename_2let = results_dir .. results_filename_base_2let
                        --local resultsFileExists_2let = paths.filep(results_filename_2let)
                        local resultsFileTooOld_2let = resultsFileExists_2let and redoResultsFilesIfOld and 
                            (lfs.attributes(results_filename_2let).modification < redoResultsIfBefore)
                                
                        if not (resultsFileExists_2let and not resultsFileTooOld_2let) then
                            showIfHave = false;
                            if haveAllFiles_2let then   -- first time we find, print this message
                                io.write(string.format('Dont have this file (and maybe others:)\n   %s \n ... have to do this experiment... \n', basename(results_filename_base_2let)))
                            elseif showIfHave then
                                io.write(string.format('  Have %s\n', basename(results_filename_base_2let)))
                            end
                            haveAllFiles_2let = false
                        else
                            --io.write(string.format('%d/%d Have %s\n', i, #allMultiLetOpt, results_filename_base_2let))
                        end
                    
                    end
                end
                
                if (resultsFileExists and not resultsFileTooOld and not redoResultsFiles)
                    and ((networkFileExists and not networkFileTooOld) or not saveNetworkToMatfile)
                    and haveAllFiles_2let then
                    io.write(string.format('      =>Already completed : %s\n', results_filename_base))
                            
                else -- if not resultsFileExists or redoResultsFiles or fileTooOld then
                    
                    local gotLock = lock.createLock(expFullTitle)
                    if not gotLock then 
                        io.write(string.format(' [Another process is already doing %s]\n', expSubtitle))
                    
                    else
                                            
                        print('Experiment subtitle : ' .. expSubtitle)
                        
                        ----- 1. LOAD DATA
                        if crowdedLetterOpts.doTextureStatistics then
                            loadOpts.normalizeInputs = true 
                        end
                        
                        local loadOpts_test_1let = table.copy(loadOpts)
                        local loadOpts_test_2let = table.copy(loadOpts)
                        loadOpts_test_1let.loadTrainingData = false
                        loadOpts_test_2let.loadTrainingData = false
                        loadOpts_test_2let.trainFrac = 0
                        
                        if not train_1let_trData then    -- load training/testing data for this font if don't have already
                            print('Loading training / test data ... ')                                                   
                                                   
                            ----- 1a. load training data
                            train_1let_trData, train_1let_tsData = loadLetters(
                                fontName1, SNR_train, crowdedLetterOpts_1let, loadOpts)
                            
                            TrainData = train_1let_trData
                            assert(train_1let_trData.nInputs  == inputStats.nInputs)
                            assert(train_1let_trData.nClasses == inputStats.nClasses)
                            assert(train_1let_trData.height   == inputStats.height)
                            assert(train_1let_trData.width    == inputStats.width)
                            
                            
                            test_1let_tsData = {}
                            test_2let_tsData = {}
            
                            ----- 1b. load test data for 1 letter (vs snr)
                            -- Load test data for 1 letter
                            for snr_i, snr in ipairs(allSNRs_test) do
                                _, test_1let_tsData[snr_i]  = loadLetters(fontName1, snr, crowdedLetterOpts_1let, loadOpts_test_1let)
                            end
                            
                    
                            ----- 1c. Load test data for multiple letters (vs nDistractors, spacing, & snr)
                            
                            if doMultiLetters then
                                test_2let_tsData = {}
                                for opt_j, multLetOpt in ipairs(allMultiLetOpt) do 
                                    test_2let_tsData[opt_j] = {}
                                    local multLetOpt_data = table.copy(multLetOpt)
                                    multLetOpt_data.trainTargetPosition = nil  -- don't append 'trained on X' to data file name
                                    --print(multLetOpt_data)
                                    for snr_i, snr in ipairs(allSNRs_test) do
                                        _, test_2let_tsData[opt_j][snr_i] = loadLetters(fontName1, snr, multLetOpt_data, loadOpts_test_2let)
                                    end        
                                end                                               
                            end
                            collectgarbage('collect')  -- train/test data can be very big -- after done with one font, 
                            collectgarbage('collect')  -- clear memory for next font data                            
                        end
                        
                        ----- 2. LOAD / TRAIN MODEL
                        
                        local pct_correct_vs_snr_any    = torch.Tensor(nSNRs):zero()
                        local pct_correct_vs_snr_target = torch.Tensor(nSNRs):zero()
                        local model_struct
                        local t_start
                        
                        trainOpts.expSubtitle = expSubtitle
                        -- trainOpts.redoTraining = false
                        model_struct = generateModel(inputStats, networkOpts_i, crowdedLetterOpts_1let)
                        model_struct = trainModel(model_struct, train_1let_trData, train_1let_tsData, trainOpts)   
                            
                        Model_struct = model_struct;
                        
                        ----- 3a. TEST Model on 1 letter (all SNRs)
                                            
                        
                        
                        io.write(string.format('*** Testing on font = %s, 1 letter \n', fontName1))   -- '  SNR = '
                        
                        local testOpts = {batchSize = trainOpts.BATCH_SIZE, nClasses = inputStats.nClasses, savePctCorrectOnIndivLetters=false, 
                            printResults = true, printInOneLine = true, returnPctCorrect = true, reshapeToVector = true}
                        
                        pct_correct_vs_snr_any = testModelOnFontsSNRs(model_struct, fontNamesList, allSNRs_test, {test_1let_tsData}, nil, testOpts)
                        
                        
                        --pct_correct_1letter_max = pct_correct_vs_snr[nSNRs]                                
                            
                        local var_list_1let = {pct_correct_vs_snr_total    = pct_correct_vs_snr_any:double(),    -- total = "all Letters"
                                               allSNRs                     = torch.DoubleTensor({allSNRs_test})} 
                                                    
                        print('  => Saved to ' .. basename(results_filename, 2))
                        
                        checkFolderExists( results_filename )
                        mattorch.save(results_filename, var_list_1let)
                        
                        if saveNetworkToMatfile then
                            print('Saving network to ' .. basename(network_matfile, 3))
                            local model_matFormat = convertNetworkToMatlabFormat(model_struct.model)
                            checkFolderExists( network_matfile )
                            mattorch.save(network_matfile, model_matFormat)
                        end
                        
                        ----- 3b. TEST Model on multiple letters (all SNRs)
                        if doMultiLetters then
                            io.write(string.format('*** Testing on multiple letters: \n'))
                                      
                            for opt_j, multLetOpt in ipairs(allMultiLetOpt) do 
                                
                                local prefix_str_a = string.format('N=%d. Spc=%2d. DNR=%.1f : pCorrA=', multLetOpt.nLetters, multLetOpt.distractorSpacing, multLetOpt.logDNR)
                                local prefix_str_t = string.rep(' ', string.find(prefix_str_a, 'pCorr')-1) .. 'pCorrT='                        
                        
                                local testOpts_any = table.copy(testOpts)
                                testOpts_any.multipleLabels = true
                                testOpts_any.doSNRheader = (opt_j == 1)
                                testOpts_any.prefix = prefix_str_a
                                
                                local testOpts_target = table.copy(testOpts)
                                testOpts_target.doSNRheader = false                                                                                        
                                testOpts_target.prefix = prefix_str_t
                                                                          
                                pct_correct_vs_snr_any    = testModelOnFontsSNRs(model_struct, fontNamesList, allSNRs_test, {test_2let_tsData[opt_j]}, nil, testOpts_any)
                                pct_correct_vs_snr_target = testModelOnFontsSNRs(model_struct, fontNamesList, allSNRs_test, {test_2let_tsData[opt_j]}, nil, testOpts_target)
                        
                                local var_list_nlet = {pct_correct_vs_snr_total        = pct_correct_vs_snr_any:double(), 
                                                       pct_correct_vs_snr_total_target = pct_correct_vs_snr_target:double(), 
                                                       allSNRs                         = torch.DoubleTensor({allSNRs_test})}
                        
                                local expSubtitle_2let = getExpSubtitle(multLetOpt, networkOpts_i, trial_i)
                                local results_filename_base_2let = stimType .. '_' .. expSubtitle_2let .. '.mat'
                                local results_filename_2let = results_dir .. results_filename_base_2let

                                print('    => Saved to ' .. basename(allMultiLetterResultsFileNames[opt_j], 2) ) --  basename(results_filename_2let, 2))
                                checkFolderExists(allMultiLetterResultsFileNames[opt_j])
                                mattorch.save(allMultiLetterResultsFileNames[opt_j], var_list_nlet)
                                    
                            end
                        end
                        lock.removeLock(expFullTitle)
                                                
                    end
            
                end  -- if don't have results file
                
            end -- loop over networks  (nNetworks)
            
            collectgarbage('collect')  -- train/test data can be very big -- after done with one font, 
            collectgarbage('collect')  -- clear memory for next font data
                        
        end  -- loop over all_LetterOpts
        
                                
    end -- loop over trials

end

   
   
   
   --[[
       local guessRate_pct = (1/13)*100
    local nRetriesIfCloseToGuessRate = 2
    local closeToGuessRate_factor = 2.5


      for try_i = 1,nRetriesIfCloseToGuessRate do
                                trainOpts.expSubtitle = expSubtitle
                                model_struct = generateModel(inputStats, networkOpts_i)
                                model_struct = trainModel(model_struct, train_1let_trData, train_1let_tsData, trainOpts)   
                                
                                
                                for si, snr in ipairs(allSNRs_test) do
                                    t_start = tic()
                                    io.write(string.format(' -- testing on font = %s, snr = %.1f ...  ', fontName, snr))
                                        
                                    local errRate_1letter = testModel_multipleLabels(model_struct, test_1let_tsData[si])
                                    pct_correct_1letter_vs_snr[si] = 100 - errRate_1letter
                                    
                                    io.write(string.format('pct correct = %.1f %% [%.2f sec]\n', pct_correct_1letter_vs_snr[si],  toc(t1)))
                                end
                                pct_correct_1letter_max = pct_correct_1letter_vs_snr[nSNRs]
                                
                                if pct_correct_1letter_max > (guessRate_pct * closeToGuessRate_factor) then
                                    break;
                                end
                                
                                print(string.format('Hm... only got pct correct = %.1f %% (vs guess rate of %.1f %%). trying again...\n', pct_correct_1letter_max, guessRate_pct))
                                trainOpts.redoTraining = true
                                
                            end
                                
    --]]
    
    
    --[[
    
                --local xrange, targetPosition, all_nDistractors, logSNR = unpack(letOpt)
            crowdedLetterOpts.all_nDistractors = toTableIfNotTable(crowdedLetterOpts.all_nDistractors)
            
            local all_nDistractors = crowdedLetterOpts.all_nDistractors
            local nnDistractors = #crowdedLetterOpts.all_nDistractors
            local nDistractorsMAX = crowdedLetterOpts.all_nDistractors[nnDistractors]
            --local allLogDNRs = crowdedLetterOpts.allLogDNRs
                             
            local crowdedLetterOpt_str = getCrowdedLetterOptsStr(crowdedLetterOpts)
            --local targetPosition_str = tostring(targetPosition)
            
            --local allDistractorSpacings_tnsr = getAllDistractorSpacings(crowdedLetterOpts.xrange, fontWidth, nDistractorsMAX)
            local allDistractorSpacings = torch.toTable(allDistractorSpacings_tnsr);
            local nSpacings = allDistractorSpacings_tnsr:numel()

--]]


                     
                        --[[
                        t_start = tic()
                        io.write('                        SNR =  ');
                        for si, snr in ipairs(allSNRs_test) do
                            io.write(string.format('  %4.1f  | ', snr))
                        end
                        io.write('\n pCorr=');
                        for si, snr in ipairs(allSNRs_test) do
                                
                            local errRate_1letter = testModel(model_struct, test_1let_tsData[si])
                            pct_correct_vs_snr_any[si] = 100 - errRate_1letter
                            
                            io.write(string.format(' %5.1f %% |', pct_correct_vs_snr_any[si]))
                        end
                        io.write(string.format(' [%.2f sec]\n', toc(t1)))
                        --]]
                        
                                                        
                                --[[
                                
                            for si, snr in ipairs(allSNRs_test) do
                                io.write(string.format('  %4.1f  | ', snr))
                            end
                            io.write('\n')
                                      

                            ...    
                                
                                io.write(prefix_str_a)
                           
                                for snr_i, snr in ipairs(allSNRs_test) do
                                    local errRate_2_i = testModel(model_struct, test_2let_tsData[opt_j][snr_i], {multipleLabels = true})
                                    
                                    local pcorrect_anyLetterOK = 100 - errRate_2_i
                                    pct_correct_vs_snr_any[snr_i] = pcorrect_anyLetterOK
                                    io.write(string.format(' %5.1f%%  |', pcorrect_anyLetterOK))
                                end        
                                io.write(string.format('\n%s', prefix_str_t))
                                
                                
                                for snr_i, snr in ipairs(allSNRs_test) do
                                    local errRate_2_i = testModel(model_struct, test_2let_tsData[opt_j][snr_i])
                                    
                                    local pcorrect_onlyTargetLetterOK = 100 - errRate_2_i
                                    pct_correct_vs_snr_target[snr_i] = pcorrect_onlyTargetLetterOK
                                    io.write(string.format(' %5.1f%%  |', pcorrect_onlyTargetLetterOK))
                                end        
                                io.write('\n');
                        --]]
                        
                        --[[
                                                    if CONVERT_IN_PARALLEL then
                                for si, snr in ipairs(SNR_train) do
                                    loadTorchDataFile(fontName1, snr, crowdedLetterOpts_1let)
                                end
                                for si, snr in ipairs(allSNRs_test) do
                                    loadTorchDataFile(fontName1, snr, crowdedLetterOpts_1let)
                                end
                                for opt_j, multLetOpt in ipairs(allMultiLetOpt) do 
                                    local multLetOpt_data = table.copy(multLetOpt)
                                    multLetOpt_data.trainTargetPosition = nil  -- don't append 'trained on X' to data file name
                                    --print(multLetOpt_data)
                                    for snr_i, snr in ipairs(allSNRs_test) do
                                        loadTorchDataFile(fontName1, snr, multLetOpt_data, multLetOpt)
                                    end        
                                end                                              
                            end
--]]