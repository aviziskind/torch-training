doCrowdedLetters = function() -- crowdedLetterOpts, loadOpts, trainOpts)
--nstates = {6,16,120} --- ORIGINAL SET ---
--nstates = {9,24,180}	  --- x 1.5 ---
--nstates = {10,27,200}
--nstates = {12,32,240}	  --- x 2 ---
--nstates = {30,80,600}	  --- x 5 --- 
--nstates = {60, 160, 1200} -- x 10 --- physiological parameters --- 
    print('Starting Crowded Letter Experiment')

    local doConvNet      = (modelName == 'ConvNet')
    local doTextureModel = (modelName == 'Texture')
    local doOverFeat     = (modelName == 'OverFeat')
        
    local netType, stimType
    if doConvNet then
        netType = 'ConvNet'
        stimType = 'NoisyLetters'
    elseif doTextureModel then
        netType = 'MLP'
        stimType = 'NoisyLettersTextureStats'
    elseif doOverFeat then        
        netType = 'MLP'
        stimType = 'NoisyLettersOverFeat'
    end
        
    local convertOverFeatFilesNow = false and (THREAD_ID  ~= nil)
    if doOverFeat and convertOverFeatFilesNow then
        print('Creating all OverFeat torch files... ')
        dofile (torchLetters_dir .. 'convertAllOverFeatFiles.lua')
        convertAllOverFeatFiles()
        return
    end
        

    local allSNRs = {0, 1, 2, 3, 4, 5, 6}
    --local allSNRs = {6}

    --local allDNRs_test = allDNRs
    local allSNRs_test = allSNRs    
    
    
    
    
    --allFontNames_ext = {'Braille', 'Sloan', 'Helvetica', 'Courier', 'Bookman', 'GeorgiaUpper', 'Yung', 'Kuenstler'}

--    local fontNames_use = {'CourierU', 'Sloan'}
--    local fontNames_use = {'CourierU', 'HelveticaUB'}
    --local fontNames_use = {'Sloan'}
    --local fontNames_use = {'HelveticaUB'}
    local fontNames_use = { 'Bookman' }
    
    --local netType = 'MLP'
        


        
        
        
    trainOpts.COST_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.TRAIN_ERR_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.MIN_EPOCHS = 20
    
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    trainOpts.REQUIRE_COST_MINIMUM = false
    loadOpts.totalUseFrac = 1
    
    --local allSizeStyles = {'sml', 'med', 'big'}
    --local allSizeStyles = {'med'}
    --local allSizeStyles = {'k18'}
    local allSizeStyles = {'k16'}
    
    if doOverFeat then
        allSizeStyles = {'k23'}
    end
    
    --xrange = letOpt[1], targetPosition = letOpt[2], nDistractors = letOpt[3], logSNR = letOpt[4],
    
    local snr_train, tbl_allSNRs_train 
    local allNetworks, allNetNames, allNetNames_nice
    
    local nTrials = 1
    
    local convFunction
    if onLaptop then
        if useGPU then
        --convFunction = 'SpatialConvolutionMap' 
            convFunction = 'SpatialConvolutionCUDA'
        else
            convFunction = 'SpatialConvolution'
        end
    else
        if useGPU then
            convFunction = 'SpatialConvolutionCUDA'
            --convFunction = 'SpatialConvolution'        
        else
            convFunction = 'SpatialConvolution'
        end

    end

        
    if netType == 'ConvNet' then
        -- ConvNet options
        

        --local allNStates = { {6,-15},  {6,-120}, {12, -120}  } --  {12, 60}, {24,120}, {48,120} } -- 
        local allNStates = { {6,-120} } --  {12, 60}, {24,120}, {48,120} } -- 
        --local allNStates = { {16,-120} } --  {12, 60}, {24,120}, {48,120} } -- 
    
        --{6,120}, {12, 240},
        --local allNStates = {  {6,30},  {6,16, 1},  } -- {6, 30} }
        --local allNStates = {  {6,120} } -- {6, 30} }
        --local allNStates = {  {24, 480} } -- {6, 30} }

        local filtSizes = {5, 4}
        --local allFiltSizes = {2, 4, 10, 20}
        
        local doPooling = true
        local allDoPooling = {false, true}
        
        local poolSize = {4, 2}
        --local filtSizes = {34}
        --local allPoolSizes = {0, 2,4,6,8,10,12,16}
        --local allPoolSizes = {0, 2,4,6,8}
        
        --local allPoolSizes = {0, 2, 4, 6}
        local allPoolSizes = {0, 2, 3, 4}
  --      
        --local allPoolSizes = {0, {2, 2}, {4, 2}, {8, 2}, {12, 2}, {16, 2}, {20, 2}, {24, 2}, {28, 2}, {32, 2}, 
  --                               {2, 4}, {4, 4}, {8, 4}, {12, 4}, {16, 4}, {20, 4} }
        --local allPoolSizes = {0, 2, {16, 4}, {20, 4} }
--        local allPoolSizes = {0, 2,4,8,16,24}
        
        local poolStride = 'auto'
        local allPoolStrides = {'auto', 2, 4, 8, 12}
        
        --local poolType = 2
        local poolType = 'MAX'
        --local allPoolTypes = {2, 'MAX'}
        

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
                                    
                                    convFunction = convFunction,
                                  }
                       
        local allNetworks_orig = expandOptionsToList(allConvNetOptions)
        
        allNetworks = fixConvNetParams(allNetworks_orig)
        
                    
        --allSNR_train = {1,2,3}
        --print(string.format('From full list of %d networks, now have %d real networks : ', #allNetworks_orig, #allNetworks) )
        --AllNetworks = allNetworks
        
        print(allNetNames)
       
       
        snr_train = {1,2,3}
        --local snr_train2 = {1,2,3,4}
        --local snr_train2 = {0, 1, 2}
        
        
        tbl_allSNRs_train = { snr_train, snr_train2 }
 
    elseif netType == 'MLP' then
        --snr_train = {2,3,4,5}
        snr_train = {1,2,3}
        
        --local allNHiddenUnits = { {6}, {12}, {24}, {48}, {96},   {6, 16}, {6, 32}, {6, 64},   {12, 16}, {12, 32}, {12, 64} }
        --local allNHiddenUnits = { 10, 30, 50, 100, 200, 500, {10, 10}, {20, 20}, {50, 50} }
       --local allNHiddenUnits = { {}, {100}, {120}, {140}, {160}, {180}, {121}, {122}, {123}, {124}, {125}, {126}, {127}, {128},{129}, {130}, {131}, {132},{133}, {134}, {135}, {136}, }
        local allNHiddenUnits = { {}, {120} }

  --          {6, 15}, {6, 30}, {6,60}, {6,120}, {6, 240}, {12, 15}, {12, 30}, {12,60}, {12,120}, {12, 240} }
        local allMLPoptions = { netType = 'MLP', 
                                tbl_nHiddenUnits = allNHiddenUnits }
                    
        allNetworks = expandOptionsToList(allMLPoptions)
        
--        allNHiddenUnits = {4,10,20,40,100,200}
        --allNHiddenUnits = {1,2, 3, 4, 5,10,25,50}
        
        local snr_train2 = {0, 1, 2}
        tbl_allSNRs_train = { snr_train, snr_train2 }
        
    end
    
    --tbl_allSNRs_train = { snr_train }
    
    --trainData, testData_contrasts = loadCrowdedLetters(fontName, fontSize, allSNRs, crowdedLetterOpts, loadOpts, true)
    
    local applyGainFactor_pink = true

    
    --local allPinkNoise_exp = {1, 1.5, 2}
    local allPinkNoise_exp = {1}
    local tbl_allPinkNoiseFilters = {}
    for i,v in ipairs(allPinkNoise_exp) do
        tbl_allPinkNoiseFilters[i] = {filterType = '1/f', f_exp = v, applyFourierMaskGainFactor = applyGainFactor_pink}
    end
    
    local whiteNoiseFilter = {filterType = 'white'}
    
    --local tbl_noiseFilters = {whiteNoiseFilter}
    local tbl_noiseFilters = table.merge(whiteNoiseFilter, tbl_allPinkNoiseFilters )
    --local tbl_noiseFilters_and_SNR = { {whiteNoiseFilter, {1, 2, 3},  table.merge(whiteNoiseFilter, tbl_allPinkNoiseFilters )
    
    local imageSize, xrange, trainTargetPosition, testTargetPosition  
    
    imageSize = {32, 160};   xrange = {-16, 12, 176};   trainTargetPosition = table.range(3,15);   testTargetPosition = 9 -- CONVNET & TEXTURE STATISTICS
    if doOverFeat then
        imageSize = {231, 231};  xrange = {-34,15,266}; trainTargetPosition = table.range(4,18);  testTargetPosition = 11   --- OVERFEAT
    end
        
    
    local all_CrowdedLetterOpts_tbl = {  
                           
                         --{ xrange = {15,5,60}, targetPosition = 'all', all_nDistractors = {1}, logSNR = 0 },
                         --{ xrange = {15,5,60}, targetPosition = 1,     all_nDistractors = {1}, logSNR = 0 },
                         --{ xrange = {15,5,60}, targetPosition = 'all', all_nDistractors = {1}, logSNR = 4 },
                         --{ xrange = {15,5,60}, targetPosition = 1,     all_nDistractors = {1}, logSNR = 5 },
                         --{ xrange = {15,30,45}, targetPosition = 'all', all_nDistractors = {1}, logSNR = 0 },
                         --{ xrange = {15,30,45}, targetPosition = 'all', all_nDistractors = {1}, logSNR = 4 },
                       
                         --{ xrange = {12,25,62}, targetPosition = 'all', all_nDistractors = {1},   logSNR = 0 }
                         --{ xrange = {12,25,62}, targetPosition = 'all', all_nDistractors = {1,2}, logSNR = 0 }
                         --{ xrange = {12,25,62}, targetPosition = 'all', all_nDistractors = {1,2}, logSNR = 0 }
                         
                         --{ xrange = {15,25,65}, targetPosition = 'all', all_nDistractors = {1,2}, logSNR = 0 }
                         
                             expName = 'Crowding', 
                             stimType = stimType,
                             tbl_fontName = fontNames_use,
                             --xrange = {15,5,55}, trainTargetPosition = 'all', 
                             --xrange = {15,5,85}, trainTargetPosition = {1,2,3,4,5,6,7,8,9}, 
                             
                             --xrange = {15,12,87}, trainTargetPosition = {1,3,4}, 
                             --xrange = {15,3,87}, trainTargetPosition = {1,9,13}, 
                             --xrange = {14,10,144}, trainTargetPosition = {1,2,3,4,5,6,7,8,9,10,11,12}, 
                             
                             --xrange = {-16,12,176}, trainTargetPosition = table.range(3,15),  imageSize = {32, 160},  -- CONVNET & TEXTURE STATISTICS
                             
                             --xrange = {-34,15,266}, trainTargetPosition = table.range(4,18), imageSize = {231, 231},   --- OVERFEAT
                             imageSize = imageSize,   xrange = xrange,  trainTargetPosition = trainTargetPosition,
                             
                             --imageSize = {32, 64}, 
                             --imageSize = {32, 128}, 
                             
                             tbl_noiseFilter = tbl_noiseFilters, 
                             doTextureStatistics = false,  Nscl_txt = 3, Nori_txt = 4, Na_txt = 5, textureStatsUse = 'V2',
                             
                             doOverFeat = doOverFeat, networkId = 0,  layerId = 19, 
                             
                             allMultiLetOpt_tbl = {  testTargetPosition = testTargetPosition,
                                                     --testTargetPosition = {11},
                                 
                                                     --tbl_nDistractors = {2},
                                                     --tbl_logDNR = {2.5} }, 
                                                     tbl_nDistractors = {1, 2},
                                                     tbl_logDNR = {2.5, 2.9} }, 
                             
                             tbl_SNR_train = tbl_allSNRs_train, 
                             tbl_sizeStyle = allSizeStyles, 
                        }
    
                    
    local all_CrowdedLetterOpts = expandOptionsToList(all_CrowdedLetterOpts_tbl)

    --All_CrowdedLetterOpts = all_CrowdedLetterOpts
    for i,letterOpts in ipairs( all_CrowdedLetterOpts ) do
   
        local nDistractorsMAX = tbl_max(letterOpts.allMultiLetOpt_tbl.tbl_nDistractors)
        local fontWidth = getFontAttrib(letterOpts.fontName, letterOpts.sizeStyle, 'width')
        local allDistractorSpacings = getAllDistractorSpacings(letterOpts.xrange, fontWidth, nDistractorsMAX, letterOpts.allMultiLetOpt_tbl.testTargetPosition)
        letterOpts.allMultiLetOpt_tbl.tbl_distractorSpacing = allDistractorSpacings
        letterOpts.allMultiLetOpt = expandOptionsToList(letterOpts.allMultiLetOpt_tbl)
        
        if type(letterOpts.fontName) == 'string' then
            letterOpts.fontName = {letterOpts.fontName}
        end
        
        for i,multLetOpt in ipairs(letterOpts.allMultiLetOpt) do
            for k,v in pairs(letterOpts) do                
                if (string.find(k, 'MultiLet') == nil) then -- don't recurse / copy 
                    multLetOpt[k] = v  -- copy all values from opt to multiletteropts
                end
            end
            multLetOpt.nLetters = multLetOpt.nDistractors + 1
        end
        
       
        
    end
        
    allNetworks, allNetNames_nice, allNetNames = uniqueNetworks(allNetworks)        
    print('\n===Using the following networks ===')
    print(allNetNames)
    print('\n===Using the following networks (full descriptions) ===')
    print(allNetNames_nice)
 
    
    --print(all_CrowdedLetterOpts)
    
    
    print('\n=== Using the following crowded-letter settings : ===');
    for i,opt in ipairs(all_CrowdedLetterOpts) do
        
        io.write(string.format(' %d : %s \n', i, getLetterOptsStr(opt)))
        for j,opt_mult in ipairs(opt.allMultiLetOpt) do
            io.write(string.format('    (%d.%2d) : %s \n', i,j,  getCrowdedLetterOptsStr(opt_mult)))
        end
        
    end
     
    --error('!')
     
    doCrowdedTrainingBatch(allNetworks, all_CrowdedLetterOpts, allSNRs_test, nTrials)
    
    
                               
   
end


