doCrowdedTrainingBatch = function(allNetworks, all_LetterOpts, allSNRs_test, nTrials)
--doCrowdedTrainingBatch(fontNames_use, allSNRs_test, allDNRs_test, allNetworks, all_LetterOpts, nTrials)    
    
    local letterSpacingPixels = 1
    local redoResultsFiles = false
    local redoResultsFilesIfOld = true
    local redoResultsIfBefore = 1390594996   -- =os.time()
           
    --local nFonts = #fontNames
    local nSNRs = #allSNRs_test
    local nLetOpts = #all_LetterOpts
    local nNetworks = #allNetworks
    
    local saveNetworkToMatfile = false --true and (expTitl ~= 'NoisyLettersTextureStats') -- and (NetType == 'ConvNet')

    local nToDo_total = nLetOpts * nNetworks * nTrials   
    local glob_idx = 1
    local curFont
       
    for trial_i = 1,nTrials do
            
        for opt_i, crowdedLetterOpts in ipairs(all_LetterOpts) do        
            
            local fontName  = crowdedLetterOpts.fontName
            local sizeStyle = crowdedLetterOpts.sizeStyle
            
            local fontWidth = getFontAttrib(fontName, sizeStyle, 'width')
            
            local SNR_train = crowdedLetterOpts.SNR_train
            local snr_train_str =  string.format('SNR-train = %s. ', toOrderedList(SNR_train))

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
                            
            local crowdedLetterOpts_1let = table.copy(crowdedLetterOpts)
            local crowdedLetterOpts_2let = table.copy(crowdedLetterOpts)
            
            crowdedLetterOpts_1let.nLetters = 1
            
            local inputStats = getDatafileStats(fontName, crowdedLetterOpts_1let)
            
            local train_1let_trData = nil
            local train_1let_tsData = nil
            
            local test_1let_tsData = nil
            local test_2let_tsData = nil
            
            io.write(string.format('============== Stim: %s (%d/%d). FONT = %s (%d/%d) ============  \n', 
                    crowdedLetterOpt_str, opt_i, #all_LetterOpts, fontName, fi, #fontNames))
           
            --print('allnetworks', allNetworks)
            for net_i, networkOpts_i in ipairs(allNetworks) do
               
                local networkStr = getNetworkStr(networkOpts_i)
                io.write(string.format('  --------- [%d / %d (%.1f%%)]   Network # %d / %d : %s :   \n', 
                        glob_idx,nToDo_total,glob_idx/nToDo_total*100,   net_i, #allNetworks, networkStr))                           
                    glob_idx = glob_idx + 1
                                    
                        
                expSubtitle = getExpSubtitle(fontName, networkOpts_i, crowdedLetterOpts, trial_i)
                
                
                expFullTitle = stimType .. '_' .. expSubtitle
                local results_filename_base = expFullTitle .. '.mat'
                local results_filename = results_dir .. results_filename_base
                local network_matfile = training_dir .. expSubtitle .. '.mat'

                local resultsFileExists = paths.filep(results_filename)
                    
                local resultsFileTooOld = resultsFileExists and redoResultsFilesIfOld and 
                        (lfs.attributes(results_filename).modification < redoResultsIfBefore)
                    
                
                local haveAllFiles_2let = true
                for ndst_i, nDistractors in ipairs(all_nDistractors) do
                    for spc_i, distractorSpacing in ipairs(allDistractorSpacings) do         
                        crowdedLetterOpts_2let.nLetters = nDistractors + 1
                        crowdedLetterOpts_2let.distractorSpacing = distractorSpacing
                        --crowdedLetterOpts_2let.logDNR = allLogDNRs[1]
                        
                        local expSubtitle_2let = getExpSubtitle(fontName, networkOpts_i, crowdedLetterOpts_2let, trial_i)
                        local results_filename_base_2let = stimType .. '_' .. expSubtitle_2let .. '.mat'
                        local results_filename_2let = results_dir .. results_filename_base_2let
                        local resultsFileExists_2let = paths.filep(results_filename_2let)
                        local resultsFileTooOld_2let = resultsFileExists and redoResultsFilesIfOld and 
                            (lfs.attributes(results_filename).modification < redoResultsIfBefore)
                            
                        if not (resultsFileExists_2let and not resultsFileTooOld_2let) then
                            if haveAllFiles_2let then   -- first time we find, print this message
                                io.write(string.format('Dont have this file (and maybe others:)\n   %s \n ... have to do this experiment... \n', results_filename_base_2let))
                            end
                            haveAllFiles_2let = false
                        end
                    end
                end
               
                            
                if resultsFileExists and not redoResultsFiles and not resultsFileTooOld and haveAllFiles_2let then
                    io.write(string.format('      =>Already completed : %s\n', results_filename_base))
                            
                else -- if not resultsFileExists or redoResultsFiles or fileTooOld then
                    
                    local gotLock = lock.createLock(expFullTitle)
                    if not gotLock then 
                        io.write(string.format(' [Another process is already doing %s]\n', expSubtitle))
                    
                    else
                                            
                        print('Experiment subtitle : ' .. expSubtitle)
                        
                        ----- 1. LOAD DATA
                        
                        local loadOpts_2let = table.copy(loadOpts)
                        
                        if not train_1let_trData then  -- load training/testing data for this font if don't have already
                            print('Loading training / test data ... ')
                        
                            ----- 1a. load training data
                            train_1let_trData, train_1let_tsData = loadLetters(
                                fontName, SNR_train, crowdedLetterOpts_1let, loadOpts, true)
                            
                            test_1let_tsData = {}
                            test_2let_tsData = {}
            
                            ----- 1b. load test data for 1 letter (vs snr)
                            -- Load test data for 1 letter
                            for snr_i, snr in ipairs(allSNRs_test) do
                                crowdedLetterOpts_1let.loadTrainingData = false
                                _, test_1let_tsData[snr_i]  = loadLetters(fontName, snr, crowdedLetterOpts_1let, loadOpts)
                            end
                            
                            ----- 1c. Load test data for multiple letters (vs nDistractors, spacing, & snr)
                            test_2let_tsData = {}
                            for ndst_i, nDistractors in ipairs(all_nDistractors) do
                                test_2let_tsData[ndst_i] = {}
                                
                                for spc_i, distractorSpacing in ipairs(allDistractorSpacings) do         
                                    test_2let_tsData[ndst_i][spc_i] = {}
                                    
                                    crowdedLetterOpts_2let.loadTrainingData = false
                                    crowdedLetterOpts_2let.targetPosition = crowdedLetterOpts_2let.testTargetPosition

                                    crowdedLetterOpts_2let.nLetters = nDistractors + 1
                                    crowdedLetterOpts_2let.distractorSpacing = distractorSpacing
                                    --crowdedLetterOpts_2let.logDNR = allLogDNRs[1]
                                    
                                    loadOpts_2let.trainFrac = 0
                                    
                                    for snr_i, snr in ipairs(allSNRs_test) do
                                        _, test_2let_tsData[ndst_i][spc_i][snr_i] = loadLetters(fontName, snr, crowdedLetterOpts_2let, loadOpts_2let)
                                    end
                                    
                                    --SampleTest = test_2let_tsData[snr_i][ndst_i][spc_i]
                                end
                                
                                
                            end
                                                
                            assert(train_1let_trData.nInputs  == inputStats.nInputs)
                            assert(train_1let_trData.nClasses == inputStats.nClasses)
                            assert(train_1let_trData.height   == inputStats.height)
                            assert(train_1let_trData.width    == inputStats.width)
                                                        
                        end

                        
                        ----- 2. LOAD / TRAIN MODEL
                        
                        local pct_correct_vs_snr = torch.Tensor(nSNRs):zero()
                        --local pct_correct_nletters_vs_snr = torch.Tensor(nSNRs):zero()
                        local model_struct
                        local t_start
                        local pct_correct_1letter_max
                        
                        trainOpts.expSubtitle = expSubtitle
                        -- trainOpts.redoTraining = false
                        model_struct = generateModel(inputStats, networkOpts_i)
                        model_struct = trainModel(model_struct, train_1let_trData, train_1let_tsData, trainOpts)   
                            
                        ----- 3a. TEST Model on 1 letter (all SNRs)
                        io.write(string.format('*** Testing on font = %s, 1 letter \n', fontName))
                        t_start = tic()
                        
                        for si, snr in ipairs(allSNRs_test) do
                            io.write(string.format(' SNR = %.1f  | ', snr))
                        end
                        io.write('\n');
                        for si, snr in ipairs(allSNRs_test) do
                                
                            local errRate_1letter = testModel_multipleLabels(model_struct, test_1let_tsData[si])
                            pct_correct_vs_snr[si] = 100 - errRate_1letter
                            
                            io.write(string.format('pCorr= %.1f %% ', pct_correct_vs_snr[si]))
                        end
                        io.write(' [%.2f sec]\n', toc(t1))
                        
                        --pct_correct_1letter_max = pct_correct_vs_snr[nSNRs]                                
                            
                        local var_list_1let = {pct_correct_vs_snr    = pct_correct_vs_snr:double(), 
                                               allSNRs               = torch.DoubleTensor({allSNRs_test})} 
                                                    
                        print('Saving results to ' .. basename(results_filename, 3))
                        mattorch.save(results_filename, var_list_1let)
                        
                        if saveNetworkToMatfile then
                            print('Saving network to ' .. basename(network_matfile, 3))
                            local model_matFormat = convertNetworkToMatlabFormat(model_struct.model)
                            mattorch.save(network_matfile, model_matFormat)
                        end

                        ----- 3b. TEST Model on multiple letters (all SNRs)
                                  
                        for ndst_i, nDistractors in ipairs(all_nDistractors) do
                            io.write(string.format('\n*** Testing on %d letters : \n             ', nDistractors+1))
                            
                            for si, snr in ipairs(allSNRs_test) do
                                io.write(string.format(' SNR = %.1f  | ', snr))
                            end
                            io.write('\n');
                            
                            --[[
                            for spc_i, distractorSpacing in ipairs(allDistractorSpacings) do
                                io.write(string.format('Spacing = %d pix.  ', distractorSpacing ))
                            end
                            io.write('\n');
                            --]]
                            for spc_i, distractorSpacing in ipairs(allDistractorSpacings) do
                                io.write(string.format('Spacing = %d pix : ', distractorSpacing ))
                            
                                crowdedLetterOpts_2let.nLetters = nDistractors + 1
                                crowdedLetterOpts_2let.distractorSpacing = distractorSpacing
                            
                            
                                local expSubtitle_2let = getExpSubtitle(fontName, networkOpts_i, crowdedLetterOpts, trial_i)
                                local results_filename_base_2let = stimType .. '_' .. expSubtitle_2let .. '.mat'
                                local results_filename_2let = results_dir .. results_filename_base_2let
                            
                            
                                for snr_i, snr in ipairs(allSNRs_test) do
                                    --io.write(string.format(' SNR = %.1f : ', snr))
                                    local errRate_2_i = testModel_multipleLabels(model_struct, test_2let_tsData[ndst_i][spc_i][snr_i])
                                    
                                    local pcorrect = 100 - errRate_2_i
                                    pct_correct_vs_snr[snr_i] = pcorrect
                                    io.write(string.format('pCorr = %.1f%%.   ', pcorrect))
                                end        
                                io.write('\n');
                                
                                local var_list_nlet = {pct_correct_vs_snr    = pct_correct_vs_snr:double(), 
                                                       allSNRs               = torch.DoubleTensor({allSNRs_test})}
                        
                                print('    => Saving results to ' .. basename(results_filename_base_2let, 3))
                                mattorch.save(results_filename_2let, var_list_nlet)
                                
                                
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


toTableIfNotTable = function(x)
    if type(x) == 'table' then
        return x
    else 
        return {x}
    end
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
    
    
    