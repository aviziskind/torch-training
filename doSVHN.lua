doSVHN = function() -- (allFontNames, allSNRs, loadOpts, noisyLetterOpts, trainOpts)
--nstates = {6,16,120} --- ORIGINAL SET ---
--nstates = {9,24,180}	  --- x 1.5 ---
--nstates = {10,27,200}
--nstates = {12,32,240}	  --- x 2 ---
--nstates = {30,80,600}	  --- x 5 --- 
--nstates = {60, 160, 1200} -- x 10 --- physiological parameters --- 
    print('Training on SVHN')
    
    expName = 'Uncertainty'
    
    if not onLaptop then
        expName = expName .. '_NYU'
    end
    --expName = 'ChannelTuning'
    
    --allFontNames_ext = {'Braille', 'Sloan', 'Helvetica', 'Courier', 'Bookman', 'GeorgiaUpper', 'Yung', 'Kuenstler'}

--    local fontNames_use = allFontNames
--    local fontNames_use = allFontNames_ext
    
    --fontNames = {'Kuenstler', 'Yung'}
    --fontNames = {'Bookman', 'GeorgiaUpper'}
    --fontNames_use = {'Bookman'}
    --fontNames_use = { {'Bookman', 'BookmanB'} }
    --fontNames_use = { {'KuenstlerU', 'KuenstlerUB'} }
    --fontNames_use = {'Bookman', {'Bookman', 'BookmanB'} }
    --fontNames_use = {'Bookman', 'Braille', 'KuenstlerU'}
    
        
    
    local netType = 'ConvNet'
    --local netType = 'MLP'
    NetType = netType

    
    trainOpts.COST_CHANGE_FRAC_THRESH = 0.01  -- = 0.1%
    trainOpts.TRAIN_ERR_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
    trainOpts.MIN_EPOCHS = 10
    trainOpts.MAX_EPOCHS = 100
    
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    trainOpts.REQUIRE_COST_MINIMUM = false
    loadOpts.totalUseFrac = 1

    
    local setName 
    setName = '' 
    --local setName = 'nStates'
    --setName = 'poolSizes'
    --setName = 'filtSizes' 
    --setName = 'filtSizes_poolSizes'
    --setName = 'LeNet'
    
    
    
    --allNHiddenUnits = {3,4,5,6,7,8,9,10,20,50,100}
    local nTrials = 1
    local snr_train, snr_train2, tbl_allSNRs_train 
    local allNetworks, allNetNames, allNetNames_nice
    
    local convFunction
    if onLaptop then
        --convFunction = 'SpatialConvolutionMap' 
        --convFunction = 'SpatialConvolutionCUDA'
        convFunction = 'SpatialConvolution'
    else
        convFunction = 'SpatialConvolutionCUDA'
        --convFunction = 'SpatialConvolution'
    end

    
    if netType == 'ConvNet' then
        -- ConvNet options

        --allNStates = { {6, -50}, {6, -100},  {12, -50}, {12, -100} }
        --allNStates = { {6, -100}  }
        --allNStates = {  {6, 16, -120}, {9,24, -180}, {12,32,-240} } -- {60, 160, -1200} }
        --allNStates = {  {6, 16, -120}, {12,32,-240}, {60, 160, -1200} } -- {60, 160, -1200} }
        
        --allNStates = { {3, 8, -60}, {6, 16, -120}, {6, 16, -30}, {12,32,-240}, } -- {60, 160, -1200} }
        --allNStates = {  {6, 16, -120}, } --  {12, 32, -120}, {24, 64, -120},  {48, 128, -120} } 
        --allNStates = {  {6, 16, -15},  {6, 16, -30}, {6, 16, -60},  {6, 16, -120}, {6, 16, -240}, {6, 16, -480}, {6, 16, -960} } -- {60, 160, -1200} }
        
        --allNStates = {  {6, 16, -120},  {6, 32, -120}, {6, 64, -120},  {6, 128, -120}, {6, 16, -960}  } -- {60, 160, -1200} }
        --allNStates = {  {3, 16, -120}, {6, 16, -120},  {12, 16, -120}, {24, 16,-120}, {48, 16, -120},   } -- {60, 160, 1200} }
        --allNStates = {  {3, 8, -60}, {6, 16, -120}, {6, 16, -30}, {6,16, -15}, {6,16,-8}, {12,32,-240},  {3,8,-10}, {3,8,-5}, {3,8,-3}, {3,8,-30}, {6,8,-10}, {12,8,-10} }
        --allNStates = {  {6, 16, -120}, {}
            
        --allNStates = { {6, 16, -960} ,
          --              {6, 4, -120},  {6, 8, -120},  {6, 16, -120},  {6, 32, -120}, {6, 64, -120},  {6, 128, -120}, {6, 256, -120},  -- {60, 160, -1200} }
            --             {3, 16, -120}, {6, 16, -120},  {12, 16, -120}, {24, 16, -120}, {48, 16, -120}, {96, 16, -120}, {192, 16, -120}  } -- {60, 160, -1200} }
            
        --local allNStates = { {6, -15}, {6, -30}, {6,-60}, {6,-120}, {6,-240}, {12, -15}, {12, -30}, {12,-60}, {12,-120}, {12, -240} }
        
        --local allNStates = { {12, -60}, {24, -120}, {48, -120} }
        
--        local allNStates = {  {16, 32, -120}, {16, -120}, }
        --local allNStates = {  {3, -15}, {6, -15}, {12, -15},  {24, -15},   {6, -30}, {6, -60}, {6, -120},   {16, -30}, {16, -60}, {16, -120}    }
        local allNStates = {  {16, -120}  }
        if hostname == 'XPS' then
            --allNStates = { {16, -120} }
        end
            
        --local allNStates = { {6,-15}, {6,-30} }
        
        --local allNStates = {12, -60}, {24,-120}, {48,-120}

        local filtSizes = {5, 4}
        --local allFiltSizes = { {5, 5, 5}}
        --local allFiltSizes = { {3, 3}, {5, 5}, }
        local allFiltSizes = {  {5, 5}, {9, 9} }
        
        local doPooling = true
        --local allDoPooling = {false, true}
        local allDoPooling = {true}

        
        --local filtSizes = {34}
        
        local poolType = 2
        local allPoolTypes = {'MAX'}  -- {1, 2, 'MAX'}
        --local allPoolTypes = {'MAX'}  -- {1, 2, 'MAX'}

        
        local poolStride = 'auto'
        --local poolStride = 2
        
        --local allPoolStrides = {'auto', 2,4,6,8}
        local allPoolStrides = {'auto'}
        --local allPoolStrides = {2,4}
            
        local poolSize = {4, 2}        
        --local allPoolSizes = { {4, 2}, {4, 4}, {2, 2}, {2, 4}, }
        --local allPoolSizes = { {4, 2}, {0, 0} } 
        --local allPoolSizes = { {4, 0, 0}, {2, 2, 0}, {0, 0, 0},  } 
        --local allPoolSizes = { 0, 2, 4, 6, 7, 8, 10, 12, 14} 
        local allPoolSizes = {0, 2, 4, 6, 8} 
        --local allPoolSizes = { {4, 0}, {4, 2}, {4, 4}, {4, 6} } 
        
        
        
        if setName == 'nStates' then
            --allNStates =  { {3,-15}, {6,-15}, {12,-15}, {24,-15},   {6,-8}, {6,-30}, {6,-60}  } 
            --allNStates =  { {6,-15}, {6,-60}, {6,-120}, {6,-240}  } 
            --allNStates =  { {6,-15},  {6,-120}, {6},  {6, 16},  {6, 16, -120},  } --{6, 16}, {6, 16,  }
            
            allNStates =  {   {6, -15}, {6, -120},   }
            
            --allNStates =  {   {6, 16},  {6, 16, -120},   {6,16, -240}, {12, 32},  {12, 32, -240}   }  -- {24, 64, -480}
             
            
        elseif setName == 'filtSizes' then
            --allFiltSizes = {2,3,4,5,10,20}
            allFiltSizes = {3,5,10}
            
        elseif setName == 'poolSizes' then
            --allDoPooling = {true}
            
            --[[
            allNStates = { { 1,15}, {1,60},  } --, {2, 30}  }
            allFiltSizes = {0}
            allPoolSizes = {0,2,4,6,8} --{0, 2,4,6,8}
            allPoolTypes = {1}
            --]]
            --allPoolSizes = {0,2,4,6,8} --{0, 2,4,6,8}
            allPoolSizes = {0,2,4,6,8} --{0, 2,4,6,8}
            
        elseif setName == 'poolTypes' then
            allPoolTypes = {1, 2, 'MAX'}
            
        elseif setName == 'filtSizes_poolSizes' then
            --allFiltSizes = {5,10}
            --allPoolSizes = {0,2,4,6}

            --allFiltSizes = {2,3,5,8,16,20,24}
            --allPoolSizes = {0, 2,4,6,8,10,12}
            
            --allFiltSizes = {2,3,5,16,24}
            --allPoolSizes = {0, 2,4,6,8,12}

            --allFiltSizes = {3,5,10,20}
            --allPoolSizes = {0, 2,4,8,16}
            
            --allNStates =  {  {6}, {6, -15}, {6, -120}, {6, -480} }, -- {6, 16}, {6, 16, -30}, {6, 16, -120},  }            
            --allNStates =  {   {6, -120},  } -- {6, 16}, {6, 16, -30}, {6, 16, -120},  }            
            --allNStates =  {  {12, -60, -60} }            
            --allNStates =  {  {6, -120} }            
            --allNStates =  {  {24, -120}, }            
            allNStates =  {  {6, -15}, }            
            --allNStates =  {  {24}, {24, -15}, {24, -120}, {24, -480}, {96}, {96, -15}, {96, -120}, {96, -480},  }            
            --allNStates =  {  {20, 50, -500}  }            
            
            allFiltSizes = { {5, 4} }
            --allFiltSizes = { {5}, {10}, {20} }
            
            --allPoolSizes = { {0, 0}, {2, 2}, {4, 2} }
            allPoolSizes = { {0, 0}, {4, 2}  } 
            --allPoolSizes = { {2, 2} } 
        elseif setName == 'LeNet' then
            
            allNStates = { {6, 16, 120, -84}, }--{6,-15} }

            allFiltSizes = { {5, 5, 5} }
            
            --local allDoPooling = {false, true}
            allDoPooling = {true, false}
                        
            allPoolStrides = {'auto'}
               
            allPoolSizes = { {2, 2, 0} } 
           
           
        end
        
        

        local allConvNetOptions = { netType = 'ConvNet', 
                                    tbl_nStates = allNStates,
                                    
                                    --filtSizes = filtSizes,
                                    tbl_filtSizes = allFiltSizes,
                                    
                                    --doPooling = doPooling, 
                                    tbl_doPooling = allDoPooling,
                                    
                                    --poolSizes = poolSize,
                                    tbl_poolSizes = allPoolSizes,
                                    
                                    --poolStrides = poolStride, 
                                    tbl_poolStrides = allPoolStrides,
                                    
                                    --poolType = poolType,
                                    tbl_poolType = allPoolTypes,
                                    
                                    --useConnectionTable = useConnectionTable,
                                    convFunction = convFunction,
                                    
                                    trainOnGPU = useGPU,
                                    
                                  }
               
       --print('print1', allNetworks)    
        local allNetworks_orig = expandOptionsToList(allConvNetOptions)
        allNetworks = fixConvNetParams(allNetworks_orig)
        --print('print2', allNetworks)
        
        
    elseif netType == 'MLP' then
        --trainOpts.MIN_EPOCHS = 3
        local snr_train1 = {6}
        local snr_train2 = {1,2,3,4}
        
        --tbl_allSNRs_train = { snr_train1, snr_train2 }
        tbl_allSNRs_train = { snr_train1 }
        
--        local allNHiddenUnits = { {6}, {12}, {24}, {48}, {96},   {6, 16}, {6, 32}, {6, 64},   {12, 16}, {12, 32}, {12, 64} }
        --local allNHiddenUnits = { {}, {4}, {5}, {8}, {10}, {15}, {30}, {60}, {120} }
--        allNHiddenUnits = {4,10,20,40,100,200}
        --allNHiddenUnits = {1,2, 3, 4, 5,10,25,50}
        
        --local allNHiddenUnits = { {}, {30}, {120}, {240}, {480}, {60, 60} }
        --local allNHiddenUnits = { {}, {30}, {120}, {240}, }
        --local allNHiddenUnits = { {}, {120} }
        local allNHiddenUnits = { {} }
            
  --          {6, 15}, {6, 30}, {6,60}, {6,120}, {6, 240}, {12, 15}, {12, 30}, {12,60}, {12,120}, {12, 240} }
        local allMLPoptions = { netType = 'MLP', 
                                tbl_nHiddenUnits = allNHiddenUnits }                    
        
        allNetworks = expandOptionsToList(allMLPoptions)
        
    end
    
    
    allNetworks, allNetNames_nice, allNetNames = uniqueNetworks(allNetworks)        
    print('\n===Using the following networks ===')
    print(allNetNames)
    print('\n===Using the following networks (full descriptions) ===')
    print(allNetNames_nice)


    local useExtraSamples = false
    
    local all_NoisyLetterOpts_tbl = { stimType = 'SVHN',
        
                                      useExtraSamples = useExtraSamples,
        
                                    }
    
    local all_NoisyLetterOpts = expandOptionsToList(all_NoisyLetterOpts_tbl)
    
    for i = 1,#all_NoisyLetterOpts do
        -- unpack values in OriXY to main struct.
        if (all_NoisyLetterOpts[i].OriXY) then   
            for k,v in pairs(all_NoisyLetterOpts[i].OriXY) do
                all_NoisyLetterOpts[i][k] = v
            end
        end     
        
        -- make sure fontName is in a table, even if just a single font.
        if type(all_NoisyLetterOpts[i].fontName) == 'string' then                            
            all_NoisyLetterOpts[i].fontName = {all_NoisyLetterOpts[i].fontName}
        end
        
    end
    
                    --[[
                all_NoisyLetterOpts = { { stimType = 'NoisyLetters', 
                                       Nori = 1, dOri = 0,    Nx = 1, dX = 0,    Ny = 1, dY = 0, 
                                       sizeStyle = 12,
                                       autoImageSize = false, 
                                       imageSize = {20, 20}, 
                                       SNR_train = snr_train
                                      } }
                              nTrials = 1
                --]]
    print('\n===Using the following fonts : ===');
    print(fontNames_use_strs)
    
    
    local noisyLetterOpts_strs = {}
    for i,opt in ipairs(all_NoisyLetterOpts) do
        noisyLetterOpts_strs[i] = getLetterOptsStr(opt) 
    end
    print('\n=== Using the following noisy-letter settings : ===');
    print(noisyLetterOpts_strs)
    
    do
        --return
    end
    
    
                      --  {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  autoImageSize = false, imageSize = {45, 45}, noiseFilter = {cycPerLet_cent = 3.2}, trainOnWhiteNoise = true } }
        
    --[[
    local all_OriXY =  { {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  blurStd = 0,  autoImageSize = false, imageSize = {50,50} },
                         {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  blurStd = 1,  autoImageSize = false, imageSize = {50,50} },
                         {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  blurStd = 1.5,  autoImageSize = false, imageSize = {50,50}},
                         {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  blurStd = 2,  autoImageSize = false, imageSize = {50,50}},
                         {stimType = 'NoisyLetters',  Nori = 1, ori_range = 0,    Nx = 1, x_range = 0,    Ny = 1, y_range = 0,  blurStd = 3,  autoImageSize = false, imageSize = {50,50}},
        --]]                 
        
                        -- imageSize = {50, 50}
                          --{1,0,  1,0,  1,0,  80,80  },
                          --{1,0,  1,0,  1,0,  20,20  },
                          --{1,0,  1,0,  1,0,  34,38  },
                        --[[{1,0,  2,3,  1,0  }, 
                        {1,0,  3,6,  1,0  }, 
                        {1,0,  7,6,  1,0  }, 
                        {1,0,  7,6,  7,6  }, 
                        {1,0,  10,9, 10,9 }, 
                        {5,8,  1,0,  1,0  }, 
                        {5,8,  5,4,  5,4  }, 
                        {5,8,  10,9, 10,9 } 
                       --]]
                                 
    doNoisyTrainingBatch(allNetworks, all_NoisyLetterOpts, allSNRs_test, nTrials)
        --function(fontNames, allSNRS_train, allSNRs_test, allNetworks, all_OriXY, allSizeStyles)
    
  --]]  
  
end


        
       
    
    
--[[

if THREAD_ID and (math.mod(fi-1, N_THREADS_TOT)+1 ~= THREAD_ID) then
                print(string.format(' ** Thread %d, skipping font # %d ** \n', THREAD_ID, fi))
                
            else
                if THREAD_ID then
                    print(string.format(' ** Thread %d, doing font # %d ** \n', THREAD_ID, fi))
                end
            
            
            --]]
            
  --[[          
  (1): nn.SpatialConvolutionMap
  (2): nn.Tanh
  (3): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> output]
    (1): nn.Square
    (2): nn.SpatialSubSampling
    (3): nn.Sqrt
  }
  (4): nn.Reshape
  
  (5): nn.Linear
  (6): nn.Tanh
  
  (7): nn.Linear
  (8): nn.LogSoftMax

--]]