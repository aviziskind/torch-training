trainModel = function(model_struct, trainData, testData, extraTrainingOpts, verbose)
    
    --print('Training Call')
    E = extraTrainingOpts
    verbose = true
    local showTrainTestTime = false
    local trainOnIndividualPositions = extraTrainingOpts.trainOnIndividualPositions
    local redoTrainingAlways = extraTrainingOpts.redoTraining --or true
    local redoTrainingIfOld = true
    local redoTrainingIfOld_date = 1402977793 -- (6/15, late) --  1402928294 --(6/15)      --1393876070 -- os.time()
    local forceContinueTraining = true
    local forceContinueTrainingIfBefore = 1417727482 -- (12/4)   1401632026 -- =(6/1) --  1399960186 -- 1399958480 --1393878677  -- os.time()
    local shuffleTrainingSamples = true -- since we're mixing sets with different noise levels, want to be evenly spread out
    local shuffleTrainingSamplesEachEpoch = true
    
    local trainingAlgorithm = 'SGD'
    --local trainingAlgorithm = 'L-BFGS'
    
    extraTrainingOpts = extraTrainingOpts or {}
    local haveTestData = testData ~= nil 
    
    local freezeFeatures = extraTrainingOpts.freezeFeatures or false-- copy this value, in case overwritten by loading from file
    local debug = false
    ---------------
    local progressBar_nStars = 50
    local continueTraining = true
    local reasonToStopTraining
    local nEpochsDone = 0
    local batchSize = extraTrainingOpts.BATCH_SIZE or 1
    local groupBatch = model_struct.parameters.convFunction and string.find(model_struct.parameters.convFunction, 'CUDA')
    local trainOnGPU = model_struct.parameters.trainOnGPU
    local prev_loss = 0
    local prev_trainErr_pct = 100
    local prev_testErr_pct
    local nClasses = trainData.nClasses
    
    if trainOnIndividualPositions then
        nClasses = nClasses * trainData.nPositions                
    end

    local trainConfusionMtx = optim.ConfusionMatrix(nClasses)
    
    
    local resizeInputToVector = model_struct.parameters.resizeInputToVector or false
    
    local expSubtitle_use = extraTrainingOpts.expSubtitle
    
    local csvLogFile   = training_dir .. expSubtitle_use .. '.csv'    
    local torchLogFile = training_dir .. expSubtitle_use .. '.t7'
        
    local trainingOpts_default = {MIN_EPOCHS = 5, MAX_EPOCHS = 500, EXTRA_EPOCHS = 0, 
        SWITCH_TO_LBFGS_AT_END = true, SAVE_TRAINING = true, 
        TEST_ERR_NEPOCHS_STOP_AFTER_MIN = 10} 
    
    local trainingOpts = trainingOpts_default
    if extraTrainingOpts then
        for k,v in pairs(extraTrainingOpts) do
            trainingOpts[k] = v
        end
    end
    
    local trainingLogger = optim.TrainingLogger(trainingOpts)    
    trainingLogger:setFile(torchLogFile)
    

    local sgd_config_default = {
        learningRate = 1e-3,
        learningRateDecay = 1e-4,
        weightDecay = 0,
        momentum = 0
    }
    local sgd_config = extraTrainingOpts.trainConfig --  or sgd_config_default;
    
    
    trainingLogger.sgd_config = sgd_config
    local lineSearchFunc, lbfgs_func 
    
    if trainOnGPU then
        lineSearchFunc = optim.lswolfe_cuda
        lbfgs_func = optim.lbfgs_cuda
    else
        lineSearchFunc = optim.lswolfe
        lbfgs_func = optim.lbfgs
    end
        
    local lbfgs_params = {
       --lineSearch = optim.lswolfe,
       lineSearch = lineSearchFunc,
       maxIter = 100,
    }
        
 
    local logger_open_mode
    local useOldFile, fileTooOld = false, false
    
    --local todate = function(t) return os.date("%a %x %X", t) end
    if redoTrainingIfOld then
        local fileDate = trainingLogger:fileDate() 
        if fileDate and fileDate < redoTrainingIfOld_date then
            print(string.format('File too old (%s < %s)', os.date("%x %X", fileDate), os.date("%x", redoTrainingIfOld_date) ) )
            fileTooOld = true
        end
        
        if not fileTooOld and fileDate and extraTrainingOpts.prevTrainingDate then
            if trainingOpts.prevTrainingDate > fileDate then   -- pre-training of network is newer than current saved training
                print(string.format('Pretrained network (%s) is NEWER than the saved training for the retrained network (%s):\nHave to start the retraining from scratch...', 
                        os.date("%x %X", trainingOpts.prevTrainingDate), os.date("%x %X", (fileDate) ) ) )
                fileTooOld = true
            else
                print(string.format('Pretrained network (%s) is older than the saved training for the retrained network (%s).\n Resuming with the retraining', 
                    os.date("%x %X", trainingOpts.prevTrainingDate), os.date("%x %X", (fileDate) ) ) ) 
            end
            
        end
        
        
    end
    
    --T = trainingLogger
    useOldFile = (not redoTrainingAlways) and trainingLogger:haveFile()  and not fileTooOld
    --print(useOldFile, trainingLogger:haveFile())

    local trainingLogger_loaded = nil
    if useOldFile then -- not redoTrainingAlways and trainingLogger:haveFile() then
        trainingLogger_loaded = trainingLogger:loadFromFile()
        
        ---[[
                    if debug then
                        T_loaded = trainingLogger_loaded
                        if T_saved then
                            print('Checking saved version from last time with just loaded version')
                            assert(isequal(T_saved, T_loaded))
                        end
                    end
        --]]
        
        if trainingLogger_loaded and not trainingLogger_loaded.sgd_config then
            trainingLogger_loaded.sgd_config = sgd_config_default
            print('Warning - no previous training configuration settings found!!')
        end
        
    end
    if trainingLogger_loaded then
        
        trainingLogger = trainingLogger_loaded
        if forceContinueTraining then
            trainingLogger:forceContinue('SGD', forceContinueTrainingIfBefore)
        end
        T = trainingLogger
        trainingLogger:setOptions(trainingOpts)
        
        ---[[
                    if debug then
                        if T_saved then
                            print('Checking saved version from last time with just loaded and force-continued version')
                            print(isequal(T_saved.model_struct, T.model_struct))
                            print(isequal(T_saved.model_struct, trainingLogger.model_struct))
                        end
                    end
        --]]
        
        nEpochsDone      = trainingLogger.nEpochs
        model_struct     = trainingLogger.model_struct
                
        prev_loss        = trainingLogger:currentLoss()
        prev_trainErr_pct = trainingLogger:currentTrainErr()
        prev_testErr_pct = trainingLogger:currentTestErr()
        
        
        io.write(string.format('   Currently at epoch %d: Cost = %.4f. TrainErr = %.2f. TestErr = %.1f\n', 
                nEpochsDone, prev_loss, prev_trainErr_pct, prev_testErr_pct))
        
        continueTraining, trainingAlgorithm, reasonToStopTraining = trainingLogger:continue()
        
        logger_open_mode = iff(paths.filep(csvLogFile), 'append', 'new')
        
        model_struct.trainingDate = trainingLogger:fileDate() 
    else
        logger_open_mode = 'new'
    end
    
    
    if (trainingLogger.filename ~= torchLogFile) then
        io.write(string.format( 'Warning! found different file name for network : \n %s \n. Changing to current name : %s\n', trainingLogger.filename, torchLogFile) )
        trainingLogger:setFile(torchLogFile)
        trainingLogger:saveToFile()
    end
    
    if not continueTraining then   
        print(string.format('Training is already complete: \n%s', reasonToStopTraining))
        return model_struct
    end
   
      -- print('Checking saved version from last time with just loaded and force-continued version, now')
    --print(isequal(T_saved, T))


    --model = model_struct.model
    local criterion = model_struct.criterion
    
    local label_field = 'labels'
    if trainOnIndividualPositions then
        label_field = 'labels_indiv_pos'
    end

    local orig_trainInputs = trainData.inputMatrix    
    local nTrainingSamples = orig_trainInputs:size(1)
    local trainingLabels  = trainData[label_field]
    
    
    local orig_testInputs, nTestSamples
    if haveTestData then
        orig_testInputs = testData.inputMatrix
        nTestSamples = orig_testInputs:size(1)    
    end



    local nTrainingSamples_full = nTrainingSamples
    local maxSizeInput_MB
    if onLaptop then
        maxSizeInput_MB = 1000 -- = ~1GB
    else
        maxSizeInput_MB = 10000 -- = ~10GB
    end
    local usePreExtractedFeatures = false
    
    local model_toTrain, model_forTest, trainInputs, trainData_use, testInputs, testData_use, feat_extractor, classifier, feat_extr_nOutputs
    MS = model_struct
    TOrig = orig_trainInputs;
    
    --- default: train the complete model using the original training/test sets.
    model_toTrain = model_struct.model
    model_forTest = model_struct.model
    
    trainInputs = orig_trainInputs
    trainData_use = trainData
    if haveTestData then
        testInputs  = orig_trainInputs
        testData_use = testData
    end
    
    
    if freezeFeatures then
        -- train only upper layers. pass inputs through lower layers to serve as new inputs
        print('Some layers are frozen : calculating new features ...')
        feat_extractor = model_struct.feat_extractor
        
        classifier = model_struct.classifier
        model_toTrain = classifier
        
        -- calculate new inputs (pass original inputs through the feature extractor)
        local sampleIdxs
        if not groupBatch then
            sampleIdxs = 1
        elseif groupBatch then
            sampleIdxs = {{1}}
        end
        
        local sampleInput = orig_trainInputs[sampleIdxs]
        local sample_outputFromFeat = feat_extractor:forward(sampleInput)     -- pass one sample through 
        feat_extr_nOutputs = torch.nElements( sample_outputFromFeat )   -- so we can see how many elements the output has
        
        local sample_size = sample_outputFromFeat:size()
        if ( #sample_size == 1 ) or (groupBatch and #sample_size == 2) then
            resizeInputToVector = true
        end                                                   
        local sizeOfNewTrainInputs_MB =  ( nTrainingSamples * feat_extr_nOutputs * 4)/(1024*1024)
        local sizeOfNewTestInputs_MB  =  ( nTestSamples     * feat_extr_nOutputs * 4)/(1024*1024)
        local sizeOfNewInputs_MB = sizeOfNewTrainInputs_MB + sizeOfNewTestInputs_MB
            
        -- double check that new features can be successfully passed to the classifier
        local sample_outputFromClass = classifier:forward(sample_outputFromFeat)
        assert( torch.nElements( sample_outputFromClass )  == nClasses)



        print(string.format('New features have %d elements (instead of the original of %d)\n', torch.nElements(sample_outputFromFeat), torch.nElements( sampleInput )  ))
        
        usePreExtractedFeatures = sizeOfNewInputs_MB < maxSizeInput_MB
        if usePreExtractedFeatures then
            print(string.format('New input training/test tensors will require %.1f MB ', sizeOfNewInputs_MB))
        else
            print(string.format('New input training/test tensor would require %.1f MB, which is greater than the max of %.1f MB. Using original inputs....', sizeOfNewInputs_MB, maxSizeInput_MB))
        end

        
        if usePreExtractedFeatures then
            model_forTest = classifier
                  
            print(string.format('Creating new training input tensor (%d x %d) (%.1f MB)',  nTrainingSamples, feat_extr_nOutputs,   sizeOfNewTrainInputs_MB ))
            local newTrainInputs = torch.Tensor(nTrainingSamples, 1, feat_extr_nOutputs):typeAs(sample_outputFromFeat)   -- if output from feat_extrator, is a cudaTensor, make new inputs are same type.
            
            progressBar.init(nTrainingSamples, 20)
            local samp_idx = {{}}
            for i = 1,nTrainingSamples do
                if not groupBatch then
                    samp_idx = i
                elseif groupBatch then
                    samp_idx[1][1] = i
                end
                newTrainInputs[i] = feat_extractor:forward(orig_trainInputs[samp_idx])
                progressBar.step()
            end        
            progressBar.done()
            NewTrainInputs = newTrainInputs
            
                            
            trainInputs = newTrainInputs
            trainData_use = {inputMatrix = trainInputs, labels = trainingLabels,      
                             nInputs = feat_extr_nOutputs, nClasses = trainData.nClasses, nPositions = trainData.nPositions} 
                
            if haveTestData then
                print(string.format('Creating new test input tensor (%d x %d) (%.1f MB)',  nTestSamples, feat_extr_nOutputs, sizeOfNewTestInputs_MB ))
                local newTestInputs = torch.Tensor(nTestSamples, 1, feat_extr_nOutputs):typeAs(sample_outputFromFeat)
                local samp_idx = {{}}
                progressBar.init(nTestSamples, 20)
                for i = 1,nTestSamples do
                    if not groupBatch then
                        samp_idx = i
                    elseif groupBatch then
                        samp_idx[1][1] = i
                    end

                    newTestInputs[i] = feat_extractor:forward(orig_testInputs[samp_idx])
                    progressBar.step()
                end
                progressBar.done()
        
                testInputs = newTestInputs
                testData_use  = {inputMatrix = testInputs, labels=testData[label_field], 
                                nInputs = feat_extr_nOutputs, nClasses = testData.nClasses, nPositions = testData.nPositions}
            end
        end
        
     
    end
    
    --Model = model_struct
    TrainData = trainData_use
    MTrain = model_toTrain
    --TrainData_use = trainData_use
    --print('GetParameters')
    local parameters, gradParameters = model_toTrain:getParameters()
        
    --print('Shuffle')
    local trainingIdxs = torch.range(1, nTrainingSamples)
    if shuffleTrainingSamples then        
        torch.manualSeed(123)
        trainingIdxs = torch.randperm(nTrainingSamples)
    end
    
    local nInputFeats = torch.nElements( trainInputs[1] )
    if freezeFeatures and not usePreExtractedFeatures then
        nInputFeats = feat_extr_nOutputs
    end


    ignoreGradient = false
    
    local checkForNan = false
    
                if debug then
                    print(string.format('first labels %d, %d, %d, %d, %d\n', trainingLabels[trainingIdxs[1]], trainingLabels[trainingIdxs[2]], 
                            trainingLabels[trainingIdxs[3]], trainingLabels[trainingIdxs[4]], trainingLabels[trainingIdxs[5]] ))
                end
    _idx_ = 0
    -- define closure to calculate cost/gradient of cost function 
    local feval = function(param_new)
                
        if parameters ~= param_new then
            parameters:copy(param_new)
        end
        gradParameters:zero()
        
        local loss = 0;
        local nThisBatch = batchSize
        local curEpoch = nEpochsDone+1
        
        if not groupBatch then
        
            for ii = 1, batchSize do
                progressBar.step()
                                
                --  idx = (idx or 0) + 1
                _idx_ = _idx_ + 1
                if _idx_ > nTrainingSamples then 
                    _idx_ = 1   -- restart index at 1
                    if shuffleTrainingSamplesEachEpoch then
                        trainingIdxs = torch.randperm(nTrainingSamples)
                    end
                end
                
                if curEpoch == 2 and _idx_ == 2 then
                    
                    --io.write(string.format('[[[E:%d]]]', nEpochsDone+1))
                end
                    
                local input = trainInputs[trainingIdxs[_idx_]]    
                if freezeFeatures and not usePreExtractedFeatures then   -- extract features now to pass the the (trainable) classifier
                    input = feat_extractor:forward(input)
                end
                local target = trainingLabels[trainingIdxs[_idx_]]  
                    
                if resizeInputToVector then           -- even if model already reshapes input, 
                    input = input:resize(nInputFeats)  -- model:backward needs vector for simple networks
                end
                Model_toTrain = model_toTrain

                local output = model_toTrain:forward(input, target)
                local nOutputs = output:numel()                
                                
                assert(output:dim() == 1)
                assert(nOutputs == nClasses)
                                            
                if output:dim() > 1 then
                    output = output:resize(nOutputs)
                end
                          
                loss = loss + criterion:forward(output, target)
                                
                model_toTrain:backward(input, criterion:backward(output, target))
                
                trainConfusionMtx:add(output,target)          

            end
            
        elseif groupBatch then
            
            _idx_ = _idx_ + 1  -- starting index
            if _idx_ > nTrainingSamples then 
                _idx_ = 1   -- restart index at 1
                if shuffleTrainingSamplesEachEpoch then
                    trainingIdxs = torch.randperm(nTrainingSamples)
                end
            end
            nThisBatch = math.min(batchSize, nTrainingSamples - _idx_ + 1)
            local batchIdxRange = { _idx_,  _idx_ + nThisBatch - 1 } 
            --local nThisBatch = batchIdxs[2] - batchIdxs[1] + 1
            
            --progressBar.step(_idx_)
            local sampleIdxs = trainingIdxs[{batchIdxRange}]
            TrainInputs = trainInputs
            --inputs = permuteBatchDimTo4(trainInputs, batchIndices)
            local inputs = selectInputBatch(trainInputs, sampleIdxs)
            Inputs = inputs
            Feat_extractor = feat_extractor
            if freezeFeatures and not usePreExtractedFeatures then
                inputs = feat_extractor:forward(inputs)
            end
            local targets = selectInputBatch(trainingLabels, sampleIdxs)

            --inputs = trainInputs[{{batchIndices}}]
            
            if resizeInputToVector then                          
                inputs = inputs:resize(nThisBatch, nInputFeats)  -- resize input appropriately
            end
            Model_toTrain = model_toTrain
            InputsNow = inputs
                        
            local outputs = model_toTrain:forward(inputs)
            assert(outputs:size(1) == nThisBatch)
            assert(outputs:size(2) == nClasses)
            
            --local input = trainInputs[trainingIdxs[_idx_]]                                    
            --local target = trainingLabels[trainingIdxs[_idx_]]  

            --if resizeInputToVector then           -- even if model already reshapes input, 
            --    input = input:resize(nInputFeats)  -- model:backward needs vector for simple networks
            --end
            
            --Model_toTrain = model_toTrain
            --local output = model_toTrain:forward(inputs, target)
                                         
            local df_do = torch.Tensor(outputs:size(1), outputs:size(2))
            for i=1,nThisBatch do
                
                loss = loss + criterion:forward(outputs[i], targets[i])
                -- estimate df/dW
                df_do[i] = criterion:backward(outputs[i], targets[i])
                trainConfusionMtx:add(outputs[i],targets[i])          

            end
            --model:backward(inputs, df_do:cuda())
                                     
            model_toTrain:backward(inputs, df_do)
            --print(_idx_, nThisBatch)
            _idx_ = _idx_ + nThisBatch-1
            
            progressBar.step(_idx_)
        
        end
       
        if nThisBatch > 1 then
            --loss = loss / nThisBatch
            --gradParameters = gradParameters:div(nThisBatch)
        end
        
        --if _idx_ <= 10 or _idx_ >= nTrainingSamples - 10 then
            --      io.write(string.format('(%d) loss = %.4f\n', idx, loss))
        --end
        
        --if ignoreGradient then
        --    gradParameters:zero()
        --end
        return loss, gradParameters
       
    end

    
    -- repeat train/test loop until stop criterion is satisfied
   
    
    local logger
    if trainingOpts.SAVE_TRAINING then
        createFolder(training_dir)
        
        logger = optim.Logger2(csvLogFile, logger_open_mode)
    end

    
    local do_SGD_Loop = function()
            
        local loss = 0
        
                if debug then
                    allLoss = torch.DoubleTensor(nTrainingSamples)
                    
                    Models_c[nEpochsDone+1] = model_toTrain:clone()
                    
                    io.write(string.format('Hashes : %.4f, %.4f, %.4f\n', networkHash(model_toTrain), networkHash(model_toTrain, 1), networkHash(model_toTrain, 1, 1)))
                end
    
        for j = 1, nTrainingSamples/batchSize do
          
          
            _,fs = optim.sgd_auto(feval,parameters,trainingLogger.sgd_config)
            --allLoss[j] = fs[1]
                       
            loss = loss + fs[1]
        end
        
                    --print(trainingLogger.sgd_config)

                if debug then
                    loss_filename = torchLetters_dir .. 'test/' .. 'SGD_epoch_' .. nEpochsDone+1 .. '__' .. getTimeStr() .. '.mat'
                    mattorch.save(loss_filename, {loss = allLoss})
                    print(string.format('\nSaved losses (mean = %.4f) to %s\n', allLoss:mean(), basename( loss_filename) ))
                    
                    --timeSpent = fevalTimer:time()
                    --print('During this loop, spent ' .. timeSpent.real .. ' seconds in feval')
                end
        
        return loss --/ nTrainingSamples
    end
    
    --error('aborted1')
    if continueTraining then
        print('Training => ' .. torchLogFile)
    end
    --print(trainingLogger.params)
                
    local current_loss, curr_trainErr_pct, fs, curr_testErr_pct
    
                if debug then
                    if T_saved then
                        print('Checking before start')
                        hash_tot_saved = networkHash(T_saved.model_struct, 1, 1)
                        hash_tot_cur = networkHash(trainingLogger.model_struct, 1, 1)
                        local tf = isequal(T_saved.model_struct, trainingLogger.model_struct)
                        local tf2 = isequal(T_saved.model_struct, model_struct)
                        local tf3 = isequal(hash_tot_saved, hash_tot_cur)
                        print(tf, tf2, tf3)
                    end
               end
   
    local checkErrBeforeStart = false
    if checkErrBeforeStart then
        curr_trainErr_pct, _, current_loss = testModel(model_forTest, trainData_use, {getLoss = true, batchSize = batchSize})        
        --curr_trainErr_pct2 = testModel(model_toTrain, trainData_use)        
        --assert(curr_trainErr_pct2 == curr_trainErr_pct)
        curr_testErr_pct = testModel(model_forTest, testData_use, {batchSize = batchSize})        
        
        io.write(string.format('   Quick check: Loss = %.4f, TrainErr = %.2f. TestErr = %.1f\n', 
                    current_loss, curr_trainErr_pct, curr_testErr_pct))
    end
    
    
                if debug then
                    if T_saved then
                        print('Checking after check err rates')
                        local tf = isequal(T_saved.model_struct, trainingLogger.model_struct)
                        local tf2 = isequal(T_saved.model_struct, model_struct)
                        print(tf, tf2)
                    end

                    print('Dummy loop');
                    --ignoreGradient = true
                    progressBar.init(nTrainingSamples, 20)
                    do_SGD_Loop()
                    progressBar.done()
                    print('Completed loop')
                    io.flush()
                    ignoreGradient = false
                end
    --]]
    
    --error('aborted')
    
    while continueTraining do
            
        io.write(string.format('Starting Epoch %d with %s: ', nEpochsDone+1, trainingAlgorithm))
        io.flush()
        
        local startTime = os.time()

        if trainingAlgorithm == 'SGD' then

            batchSize = extraTrainingOpts.BATCH_SIZE or 1
            progressBar.init(nTrainingSamples, progressBar_nStars) --- progressbar for SGD
        
            trainConfusionMtx:zero()
                            
            local startTime = os.time()
            current_loss = do_SGD_Loop()
            progressBar.done()
                   
            trainConfusionMtx:updateValids()
            curr_trainErr_pct = (1-trainConfusionMtx.totalValid)*100   --trainError = testModel(model, testData)        

        elseif (trainingAlgorithm == 'L-BFGS') or (trainingAlgorithm == 'L-BFGS-reduced') then
            if (trainingAlgorithm == 'L-BFGS-reduced') then
                nTrainingSamples = math.floor(nTrainingSamples_full * trainingOpts.LBFGS_REDUCED_FRAC)

            elseif (trainingAlgorithm == 'L-BFGS') then
                nTrainingSamples = nTrainingSamples_full    
            end
            batchSize = nTrainingSamples
            --fevalTimer:reset()
            progressBar.init(nTrainingSamples, progressBar_nStars, '=', true)
                        
            --_, fs = optim.lbfgs_cuda(feval, parameters, lbfgs_params)
            --_, fs = optim.lbfgs(feval, parameters, lbfgs_params)
            _, fs = lbfgs_func(feval, parameters, lbfgs_params)
            current_loss = fs[1]
    
            curr_trainErr_pct = testModel(model_forTest, trainData_use, {verbose = showTrainTestTime, test_indiv_pos = trainOnIndividualPositions, batchSize = batchSize}) -- L-BFGS does multiple loops
                
        end
       
        local timeForThisEpoch = os.time() - startTime
        if showTrainTestTime and (trainingAlgorithm == 'SGD') then
            local timeEachSample_sec = (timeForThisEpoch / nTrainingSamples)
            print('==> time to train each sample = ' .. timeEachSample_sec *1000 .. 'ms')
        end


       
        nEpochsDone = nEpochsDone + 1;
        
        local loss_change_pct = (current_loss - prev_loss)/prev_loss * 100
        prev_loss = current_loss;
 
        local t_elapsed_sec = 0
        local trainErr_pct_change_pct = (curr_trainErr_pct - prev_trainErr_pct)/prev_trainErr_pct*100 
        prev_trainErr_pct = curr_trainErr_pct
        
        if haveTestData then
            curr_testErr_pct, _, _, t_elapsed_sec = 
                testModel(model_forTest, testData_use, {verbose = showTrainTestTime, test_indiv_pos = trainOnIndividualPositions, batchSize = batchSize})        
        else
            curr_testErr_pct = 0
        end
        
        --io.write('[print]')
        
        --io.write('[add\n');
        
        trainingLogger:add(nEpochsDone, model_struct, current_loss, curr_trainErr_pct, curr_testErr_pct, timeForThisEpoch)
        --io.write(']')
        
        
        --io.write('[Cont')
        continueTraining, trainingAlgorithm, reasonToStopTraining = trainingLogger:continue()
        --io.write(']')
        
        
        local checkErrAfterEachEpoch = false
        if checkErrAfterEachEpoch then
            local curr_trainErr_pct_after, _, current_loss_after = testModel(model_forTest, trainData_use, {getLoss = true, batchSize = batchSize})
            --local curr_trainErr_pct_after2 = testModel(model_toTrain, trainData_use)
            --assert(curr_trainErr_pct_after == curr_trainErr_pct_after2)
            local curr_testErr_pct_after = testModel(model_forTest, testData_use, {batchSize = batchSize})        
            
            io.write(string.format('   Quick check: Loss = %.4f. TrainErr = %.2f. TestErr = %.1f\n', 
                        current_loss_after, curr_trainErr_pct_after, curr_testErr_pct_after))
        end

        
        if trainingOpts.SAVE_TRAINING then
            tic()
            io.write(string.format('[Saving...' ))
            trainingLogger:saveToFile()
            local t_elapsed = toc()
            io.write(string.format('done:%s]', sec2hms(t_elapsed)))
            
            logger:add{['Epoch'] = nEpochsDone}        
            logger:add{['Cost'] = current_loss}
            logger:add{['Train Err'] = curr_trainErr_pct}
            logger:add{['Test Err'] = curr_testErr_pct}
            logger:add{['Time'] = timeForThisEpoch}
            logger:println()
            
            if debug then
                --T_saved = table.copy(trainingLogger)
                --Model_struct = model_struct
                io.write(string.format('Hashes : %.4f, %.4f, %.4f\n', networkHash(model_struct.model), networkHash(model_struct.model, 1), networkHash(model_struct.model, 1, 1)))
            end

        end
        model_struct.trainingDate = os.time()

        local timeForThisEpoch_tot = os.time() - startTime
        io.write(string.format('   after epoch %d: Cost = %.4f (%+.2f%%). TrainErr = %.2f (%+.2f%%). TestErr = %.1f [took %s]\n\n', 
                nEpochsDone, current_loss, loss_change_pct, curr_trainErr_pct, trainErr_pct_change_pct, curr_testErr_pct, sec2hms(timeForThisEpoch_tot)))


        --error('!')
        --torch.save(networks_dir..'network_'.. epoch .. '.bin', model)
        do
            --error('!')
        end
        if not continueTraining then
            io.write(string.format('Stopped training : %s\n', reasonToStopTraining))
        end
   
    end

    return model_struct   -- important to return this at the end -- in case have loaded saved file and are replacing original one

    
end






--testModel = function(model_struct, testData, verbose, getLoss)
testModel = function(model_struct, testData, opt)

    opt = opt or {}
    local verbose = opt.verbose or false
    local getLoss = opt.getLoss or false
    local multipleLabels = opt.multipleLabels or false
    local test_indiv_pos = opt.test_indiv_pos or false
    local nClasses = testData.nClasses
    local returnPctCorrect = opt.returnPctCorrect or false 
    

    tic()
    local model
    if (model_struct.modelType ~= nil) then
        --print('test mode 1')
        model = model_struct.model
    else
        --print('test mode 2')
        model = model_struct
    end
    local groupBatch = areAnySubModulesOfType(model, 'nn.SpatialConvolutionCUDA') 
    local batchSize = 1
    if (opt.batchSize and groupBatch) then
        batchSize = opt.batchSize
    end

    
    local useConfusionMatrix = true and not multipleLabels
    
	local testInputs = testData.inputMatrix

    local labels_field = 'labels'
    if test_indiv_pos then
        labels_field = 'labels_indiv_pos'
        nClasses = nClasses * testData.nPositions
    end
	local labels = testData[labels_field]
    local haveSecondLabels = testData.labels_distract ~= nil   -- for testing with multiple labels
    local haveThirdLabels  = testData.labels_distract2 ~= nil

    local testConfusionMtx
    if useConfusionMatrix then
        testConfusionMtx = optim.ConfusionMatrix(nClasses)
        testConfusionMtx:zero()
    end

    
    --testInputs_copy
	
    if verbose then
        --print('==> testing ')
    end
    
	local nTestSamples = testInputs:size(1)
    local loss = 0
    local nCorrect = 0
    local criterion
    local target, idx_max
    if getLoss then
        if model_struct.criterion then
            criterion = model_struct.criterion
        else
            criterion = nn.ClassNLLCriterion();
        end
    end
    
    TestInputs = testInputs
    Labels = labels
    M = model
    
    
    local nBatches = math.ceil(nTestSamples / batchSize)
    local pred
    
	for batch_i = 1, nBatches do

        
        local idxs = { (batch_i-1) * batchSize +1,  math.min(batch_i * batchSize, nTestSamples) } 
        local nThisBatch = idxs[2] - idxs[1] + 1
        
        local preds, targets
        
        if groupBatch then        
            preds = model:forward(testInputs[{ idxs }])
            --Preds = M:forward(TestInputs[{ idxs }])
            targets = labels[{idxs}] 
        else
            preds = model:forward(testInputs[{ idxs[1] }])
            targets = labels[{idxs[1]}] 
        end
        
        
        for samp_i = 1, nThisBatch do
            if preds:dim() == 1 then
            --if groupBatch and nThisBatch > 1 then
                pred = preds
            else 
                pred = preds[samp_i]
            end
            if type(targets) == 'number' then
                target = targets
            else
                target = targets[samp_i]
            end            
        
            if useConfusionMatrix then
                --Pred = pred
                --Target = target
                --CMtx = testConfusionMtx
                testConfusionMtx:add(pred,target)
            
            else

                local idx_sample  = idxs[1] + samp_i - 1
                _,idx_max = torch.max(pred:resize(nClasses), 1)
                --local _,idx_max = torch.max(pred, 1)

                if idx_max[1] == testData.labels[idx_sample] then
                    nCorrect = nCorrect + 1
                elseif multipleLabels and
                    (haveSecondLabels and (idx_max[1] == testData.labels_distract[idx_sample])) or
                    (haveThirdLabels  and (idx_max[1] == testData.labels_distract2[idx_sample])) then
                        nCorrect = nCorrect + 1
                end
            
            end
            
            if getLoss then
                loss = loss + criterion:forward(pred, target)
            end
            
        end
        
	end
    
    
    
    local testErr_pct_total,     testCorrect_pct
	local testErr_pct_eachClass, testCorrect_pct_eachClass
    
    if useConfusionMatrix then
        testConfusionMtx:updateValids()

        local fracCorrect_total = testConfusionMtx.totalValid -- a number
        local fracCorrect_eachClass = testConfusionMtx.valids -- a tensor
        
        testErr_pct_total     = (1 - fracCorrect_total)*100        -- a number
        testErr_pct_eachClass = (- fracCorrect_eachClass + 1)*100  -- a tensor

    else
    
        testErr_pct_total = (nTestSamples - nCorrect) / nTestSamples * 100
        testErr_pct_eachClass = torch.Tensor(nClasses):zero()
    
    end
    

    if getLoss then
        loss = loss / nTestSamples
    end


    if verbose then
        local totalTime = toc()
        local timeEachSample_sec = (totalTime / nTestSamples)
        io.write(string.format('\n [con]: %.2f ==> Total time: %.2f sec. Time for each sample = %.2f ms. ', 
                testErr_pct_total, totalTime, timeEachSample_sec *1000))
        --print(testConfusionMtx)
    end

    local t_elapsed = toc()
        
    
    if returnPctCorrect then
        local testCorrect_pct     = 100 - testErr_pct_total            -- a number
        testCorrect_pct_eachClass =      -testErr_pct_eachClass + 100  -- a tensor (so need to put first to add a number)
        return testCorrect_pct, testCorrect_pct_eachClass, loss, t_elapsed
    else

        return testErr_pct_total, testErr_pct_eachClass, loss, t_elapsed
    end

end


--]]


getTimeStr = function()
    return os.date("%H_%M_%S")
    --[[
    local s = os.date()
    local _, st = string.find(s, '2015 ')
    local en = string.find(s, ' %uM %u%uT')  -- e.g. AM EDT, PM EST, etc.
    s = string.gsub( string.gsub( string.sub(s, st+1, en-1), ':', '_'), ' ', '_')    
    return s
    -- Mon 02 Jun 2014 03:56:20 PM EDT
    --]]
end


permuteBatchDimTo4 = function(x, x_indices)
--    print('!!')
    local sizeX = x:size()
    
    if #sizeX < 4 then
        error('Not enough dimensions')
    end
    local nSamplesTot, nPlanes, h, w = sizeX[1], sizeX[2], sizeX[3], sizeX[4]
    
    x_indices = x_indices or torch.range(1, nSamplesTot)
    local batchSize = x_indices:numel()
    
    local y = torch.Tensor(nPlanes, h, w, batchSize)
    for sample_i = 1, batchSize do
        y[{ {}, {}, {}, sample_i }] = x[{  x_indices[sample_i]  }]
    end
    
    return y
    
end


selectInputBatch = function(x, x_indices)
    
    local sizeX = x:size()
    
    local nSamplesTot = sizeX[1]
    local batchSize = x_indices:numel()
    
    local sizeY = sizeX  
    sizeY[1] = batchSize  
    y = torch.Tensor(sizeY):typeAs(x)
    
    for sample_i = 1, batchSize do
        y[{ sample_i }] = x[{  x_indices[sample_i]  }]
    end
        
    return y
    
end


--logger--------------------------
-----------------------------------

--[[
getPercentCorrect = function()
    nCorrect = 0
    for i = 1, m do
        local input_1 = inputs[i]
        local target_1 = targets[i][1]

        hyp_1 = model:forward(input_1, target_1)
        _,idx_max = torch.max(hyp_1:resize(nClasses), 1)

        if idx_max[1] == target_1 then
            nCorrect = nCorrect + 1
        end        
    end
    return nCorrect / m
end
--]]

--[[
if opt.optimization == 'CG' then
    config = config or {maxIter = opt.maxIter}
    optim.cg(feval, parameters, config)

elseif opt.optimization == 'LBFGS' then
    config = config or {learningRate = opt.learningRate,
                        maxIter = opt.maxIter,
                        nCorrection = 10}
    optim.lbfgs(feval, parameters, config)
elseif opt.optimization == 'SGD' then
    config = config or {learningRate = opt.learningRate,
                        weightDecay = opt.weightDecay,
                        momentum = opt.momentum,
                        learningRateDecay = 5e-7}
    optim.sgd(feval, parameters, config)
elseif opt.optimization == 'ASGD' then
    config = config or {eta0 = opt.learningRate,
                        t0 = trsize * opt.t0}
    _,_,average = optim.asgd(feval, parameters,config)

else
    error('unkown optimization method')
end
        --]]
        
        
        --[[
        
testModel_multipleLabels = function(model_struct, testData, verbose)
    
    verbose = verbose or false

    local model
    if (model_struct.modelType ~= nil) then
        model = model_struct.model
    else
        model = model_struct
    end
    
    local nClasses = testData.nClasses
	local testInputs = testData.inputMatrix
	--local labels = testData.labels
    local haveSecondLabels = testData.labels_distract ~= nil
    local haveThirdLabels = testData.labels_distract2 ~= nil
    --testData_copy = testData
    --testInputs_copy
    
   
    
	local nTestSamples = testInputs:size(1)
    --model_copy = model
    --testInputs_copy = testInputs
    --labels_copy = labels
    
    local nCorrect = 0
	for t = 1, nTestSamples do

		local pred = model:forward(testInputs[t])
		
        --pred = model:forward(testInputs[t], labels[t])
        local _,idx_max = torch.max(pred:resize(nClasses), 1)
        --local _,idx_max = torch.max(pred, 1)

        if idx_max[1] == testData.labels[t] then
            nCorrect = nCorrect + 1
        elseif haveSecondLabels and idx_max[1] == testData.labels_distract[t] then
            nCorrect = nCorrect + 1
        elseif haveThirdLabels and idx_max[1] == testData.labels_distract2[t] then        
            nCorrect = nCorrect + 1
        end
        
    end
    
    local testErr_pct = (nTestSamples - nCorrect) / nTestSamples * 100
        

    if verbose then
        local totalTime = toc()
        local timeEachSample_sec = (totalTime / nTestSamples)
        io.write(string.format('\n [sim]: %.2f ==> Total time: %.2f sec. Time for each sample = %.2f ms. ',
                testErr_pct, totalTime, timeEachSample_sec *1000))
        --print(testConfusionMtx)
    end

    return testErr_pct   
   

end
--]]




--[[

train new - insert currentDate 
retrain new  

train - load
retrain new

train - load get date of training
retrain - load *** only load if newer than previous training***


train new (continued or something)
retrain load

--]]

--[[

             --local let = function(i) return string.char(i+64) end
                
                local idx_sample  = idxs[1] + samp_i - 1
                _,idx_max = torch.max(pred:resize(nClasses), 1)
                --local _,idx_max = torch.max(pred, 1)

                local isCorrect = false
                
                if idx_max[1] == testData.labels[idx_sample] then
                    
                    isCorrect = true
                elseif multipleLabels and
                    (haveSecondLabels and (idx_max[1] == testData.labels_distract[idx_sample])) or
                    (haveThirdLabels  and (idx_max[1] == testData.labels_distract2[idx_sample])) then
                    
                    isCorrect = true
                end
                
                if isCorrect then
                    nCorrect = nCorrect + 1
                    
                    if idx_sample < 8 then
                        io.write('%s:%s%s%s', let(idx_max[1]), let(testData.labels[t])
                        
                        io.write('[Y]')
                    else
                        io.write('[N]')
                    end
                    
                end
--]]
            
            
            