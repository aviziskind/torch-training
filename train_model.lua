trainModel = function(model_struct, trainData, testData, trainingOpts, verbose)
    
    --print('Training Call')
    E = trainingOpts
    verbose = true
    local showTrainTestTime = false
    local trainOnIndividualPositions = trainingOpts.trainOnIndividualPositions
    local redoTrainingAlways = trainingOpts.redoTraining or false
    local redoTrainingIfOld = true
    --local redoTrainingIfOld_date = 1444887467 
    local redoTrainingIfOld_date = 1402977793 -- (6/15, late) --  1402928294 --(6/15)      --1393876070 -- os.time()
    local forceContinueTraining = true
    local forceContinueTrainingIfBefore = 1417727482 --  (12/4)   1401632026 -- =(6/1) --  1399960186 -- 1399958480 --1393878677  -- os.time()
    local shuffleTrainingSamples = true -- since we're mixing sets with different noise levels, want to be evenly spread out
    local shuffleTrainingSamplesEachEpoch = true
    
    if not shuffleTrainingSamples then
        print(' Warning - shuffling training samples is turned OFF !! ');
    end
    local trainingAlgorithm = 'SGD'
    --local trainingAlgorithm = 'L-BFGS'
    
    trainingOpts = trainingOpts or {}
    local haveTestData = testData ~= nil 
    local haveSecondTestData = trainingOpts.test2Data ~= nil
    local test2Data = trainingOpts.test2Data
    if haveSecondTestData then
        print('>> Have a second test set << ')
    end
    
    if trainingOpts.redoTrainingIfOld_date then
        redoTrainingIfOld_date = trainingOpts.redoTrainingIfOld_date
    end
    
    local freezeFeatures = trainingOpts.freezeFeatures or false-- copy this value, in case overwritten by loading from file
    local debug = false
    ---------------
    local progressBar_nStars = 50
    local continueTraining = true
    local reasonToStopTraining
    local epoch_id = 1
    local batchSize = trainingOpts.BATCH_SIZE or 1
    local groupBatch = model_struct.parameters.convFunction and string.find(model_struct.parameters.convFunction, 'CUDA')
    local trainOnGPU = model_struct.parameters.trainOnGPU or trainingOpts.trainOnGPU
    local nOutputs = trainData.nClasses  or trainData.nOutputs
    
    local memoryAvailableBuffer_MB = trainingOpts.memoryAvailableBuffer or 3000 --- dont leave less than this amount of memory available
    
    local criterion = model_struct.criterion
    local trainingClassifier = torch.isClassifierCriterion(torch.typename(criterion))
    trainingOpts.trainingClassifier = trainingClassifier
    
    local showCost = true
    local showErr  = trainingClassifier
    
    --local haveExtraCriterion = model_struct.criterion2
    local origCriterion = model_struct.criterion
    local extraCriterion = model_struct.extraCriterion
    if extraCriterion then
        print('origCriterion', origCriterion)
        print('extraCriterion', extraCriterion)
    end
    EC = extraCriterion

    criterion_firstEpoch = model_struct.criterion_firstEpoch
    local nEpochsFirstCriterion = 1
    if criterion_firstEpoch then
        if criterion_firstEpoch.nEpochs then
            nEpochsFirstCriterion = criterion_firstEpoch.nEpochs
        end
        if criterion_firstEpoch.nEpochsTitration then
            nEpochsFirstCriterion = nEpochsFirstCriterion + criterion_firstEpoch.nEpochsTitration
        end
        cprintf.yellow('criterion for first %d epoch(s) = %s\n', nEpochsFirstCriterion, criterion_firstEpoch)
    end
    
    --local MSEWeightsDependOnTarget =  (torch.typename(criterion) == 'nn.WeightedMSECriterion') and criterion.MSEWeightsDependOnTarget
    
    
    local require_cost_minimum = false
    if not trainingClassifier then
        require_cost_minimum = true
    end
    
    
    local reprocessInputs = trainingOpts.reprocessInputs
    print( ('Have reprocessInputs = ' .. tostring(reprocessInputs ~= nil) ) )
    
    local reprocessInputsAndTargets = trainingOpts.reprocessInputsAndTargets
    print( ('Have reprocessInputsAndTargets = ' .. tostring(reprocessInputsAndTargets ~= nil) ) )
    
    local targetsDependOnEpoch = trainingOpts.targetFunction ~= nil
    local targetFunction = trainingOpts.targetFunction
    if targetsDependOnEpoch then
        cprintf.Red('Have targetFunction \n')
    end

    local useFunctionToGetTrainingIdx = trainingOpts.getTrainingIdx ~= nil
    local getTrainingIdx = trainingOpts.getTrainingIdx

    
    if trainOnIndividualPositions then
        nOutputs = nOutputs * trainData.nPositions                
    end

    local trainConfusionMtx = optim.ConfusionMatrix(nOutputs)
    
    local resizeInputToVector = false --model_struct.parameters.resizeInputToVector or false
    
    local trainingFileBase = trainingOpts.trainingFileBase
    
    local csvLogFile   = trainingFileBase .. '.csv'    
    local torchLogFile = trainingFileBase .. '.t7'
    local trainingDir = paths.dirname(trainingFileBase)
        
    if redoTrainingAlways then
        io.write('\n === Warning -- retraining ALL models ===\n')
    end 
        
    --local trainingOpts_default = {MIN_EPOCHS = 1000, MAX_EPOCHS = 2000, EXTRA_EPOCHS = 0, 
    local trainingOpts_default = {MIN_EPOCHS = 2, MAX_EPOCHS = 50, EXTRA_EPOCHS = 0, 
        SWITCH_TO_LBFGS_AT_END = false, SAVE_TRAINING = true, 
        TEST_ERR_NEPOCHS_STOP_AFTER_MIN = 10, REQUIRE_COST_MINIMUM = require_cost_minimum} 
    
    --print(trainingOpts_default)
    for k,v in pairs(trainingOpts_default) do
        if not trainingOpts[k] then
            trainingOpts[k] = v
        end
    end
    
    local trainingLogger = optim.TrainingLogger(trainingOpts)    
    trainingLogger:setFile(torchLogFile)
    
        
    --local curEpochIdx = 1

    local sgd_config_default = {
        learningRate = 1e-3,
        learningRateDecay = 1e-4,
        weightDecay = 0,
        momentum = 0
    }
    local sgd_config = trainingOpts.trainConfig --  or sgd_config_default;
    
    
    trainingLogger.sgd_config = sgd_config
    print(trainingLogger.sgd_config)
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
        
        if not fileTooOld and fileDate and trainingOpts.prevTrainingDate then
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
    
    TL = trainingLogger
    useOldFile = (not redoTrainingAlways) and trainingLogger:haveFile()  and not fileTooOld
    --print(useOldFile, trainingLogger:haveFile())

    local trainingLogger_loaded = nil
    if useOldFile then -- not redoTrainingAlways and trainingLogger:haveFile() then
        trainingLogger_loaded = trainingLogger:loadFromFile()
        
        
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
        --]]
        
        epoch_id      = trainingLogger.nEpochs + 1
        model_struct     = trainingLogger.model_struct
        if model_struct.sampleJointErrors and useFunctionToGetTrainingIdx then
            getTrainingIdx(model_struct.sampleJointErrors)
        end
        
                
        local prev_train_loss   = trainingLogger:currentLoss()
        local prev_trainErr_pct = trainingLogger:currentTrainErr()
        local prev_testErr_pct = trainingLogger:currentTestErr()
        
        
        io.write(string.format('   Currently at epoch %d: Cost = %.4f.', epoch_id, prev_train_loss));
        if showErr then
            io.write(string.format('TrainErr = %.3f. TestErr = %.3f', prev_trainErr_pct, prev_testErr_pct))
        end
        io.write('\n')
        
        continueTraining, trainingAlgorithm, reasonToStopTraining = trainingLogger:continue()
        
        
        if continueTraining and trainingOpts.shouldBeCompletedTraining then
            print(continueTraining, trainingAlgorithm, reasonToStopTraining)
            error('Should have completed training!!')
        end
        
        
        logger_open_mode = iff(paths.filep(csvLogFile), 'append', 'new')
        
        model_struct.trainingDate = trainingLogger:fileDate() 
    else
        logger_open_mode = 'new'
    end
    
    print('trainOnGPU', trainOnGPU)
    local modelToUse = model_struct.modelToUse or model_struct.model
    
    if trainOnGPU then
        modelToUse:cuda()
        model_struct.criterion:cuda()
    else
        modelToUse:float()
        model_struct.criterion:float()
    end
    
    if (trainingLogger.filename ~= torchLogFile) then
        io.write(string.format( 'Warning! found different file name for network : \n %s \n. Changing to current name : %s\n', trainingLogger.filename, torchLogFile) )
        trainingLogger:setFile(torchLogFile)
        trainingLogger:saveToFile()
    end
    
    if not continueTraining then   
        print(string.format('Training is already complete: \n%s', reasonToStopTraining))
        return model_struct
    else
        print(string.format('Training is not yet complete: \n%s', reasonToStopTraining))
    end
   
      -- print('Checking saved version from last time with just loaded and force-continued version, now')
    --print(isequal(T_saved, T))


    --model = model_struct.model
   
    
    local output_field 
    if trainingClassifier then
        output_field = 'labels'
        --output_field = 'outputMatrix'
        if trainOnIndividualPositions then
            output_field = 'labels_indiv_pos'
        end
    elseif model_struct.outputTargetField then
        output_field = model_struct.outputTargetField
    else
        output_field = 'outputMatrix'
    end
    local network_sampleOutput
    

    local input_field = model_struct.inputDataField or 'inputMatrix'
    
    local orig_trainInputs = trainData[input_field]
        
    local size1 = function(x)
        if torch.isTensor(x) then 
            return x:size(1)
        elseif type(x) == 'table' then 
            return #x
        end
    end
        
            
    local weightsFunctionEpoch
    local nTrainingSamples = size1(orig_trainInputs)
    local trainingOutputs  = trainData[output_field]
    
    TrainData = trainData
    F = output_field
    
    local orig_testInputs, nTestSamples, testingOutputs
    if haveTestData then

        orig_testInputs = testData[input_field]
        nTestSamples = size1(orig_testInputs)
        testingOutputs = testData[output_field]
    end



    local nTrainingSamples_full = nTrainingSamples
    local maxSizeInput_MB
    if onLaptop then
        maxSizeInput_MB = 2000 -- = ~1GB
    else
        maxSizeInput_MB = 25000 -- = ~10GB
    end
    local usePreExtractedFeatures = false
    
    local model_toTrain, model_struct_forTest, model_forTest, trainInputs, trainData_use, testInputs, testData_use, feat_extractor, classifier, feat_extr_nOutputs
    MS = model_struct
    TOrig = orig_trainInputs;
    
    --- default: train the complete model using the original training/test sets.
    model_toTrain = modelToUse
    model_forTest = modelToUse
    
    
    trainInputs = orig_trainInputs
    TrainInputs = trainInputs
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
        Smp = sample_outputFromFeat
        
        local feat_extr_outputSize = sample_outputFromFeat:size()
        if ( #feat_extr_outputSize == 1 ) or (groupBatch and #feat_extr_outputSize == 2) then
            resizeInputToVector = true
        end                                                   
        local sizeOfNewTrainInputs_MB =  ( nTrainingSamples * feat_extr_nOutputs * 4)/(1024*1024)
        local sizeOfNewTestInputs_MB  =  ( nTestSamples     * feat_extr_nOutputs * 4)/(1024*1024)
        local sizeOfNewInputs_MB = sizeOfNewTrainInputs_MB + sizeOfNewTestInputs_MB
            
        -- double check that new features can be successfully passed to the classifier
        local sample_outputFromClass = classifier:forward(sample_outputFromFeat)
        assert( torch.nElements( sample_outputFromClass )  == nOutputs)


        print(string.format('New features have %d elements (instead of the original of %d)\n', torch.nElements(sample_outputFromFeat), torch.nElements( sampleInput )  ))
        
        local memoryAvailable_MB = sys.memoryAvailable()
        if sizeOfNewInputs_MB > (memoryAvailable_MB  - memoryAvailableBuffer_MB ) then
            error(string.format('Not enough memory available. New inputs would take up %.1f MB, but only have %.1f MB available (and want to have a buffer of %.1f\n Free up some memory for this experiment.', sizeOfNewInputs_MB, memoryAvailable_MB, memoryAvailableBuffer_MB))
        end
        
        usePreExtractedFeatures = sizeOfNewInputs_MB < maxSizeInput_MB
        
        if usePreExtractedFeatures then
            print(string.format('New input training/test tensors will require %.1f MB ', sizeOfNewInputs_MB))
        else
            print(string.format('New input training/test tensor would require %.1f MB, which is greater than the max of %.1f MB. Using original inputs....', sizeOfNewInputs_MB, maxSizeInput_MB))
        end

        
        if usePreExtractedFeatures then
            model_forTest = classifier
                  
            print(string.format('Creating new training data : %d samples of size %s (%.1f MB)', 
                    nTrainingSamples, toList( feat_extr_outputSize, nil, ' x '),  sizeOfNewTrainInputs_MB ))
            local newInputSize = torch.concat(nTrainingSamples, feat_extr_outputSize)
            local newTrainInputs = torch.Tensor(newInputSize):typeAs(sample_outputFromFeat)   -- if output from feat_extrator, is a cudaTensor, make new inputs are same type.
            
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
            trainData_use = {[input_field] = trainInputs, [output_field] = trainingOutputs,      
                             nInputs = feat_extr_nOutputs, nOutputs = nOutputs, nPositions = trainData.nPositions} 
                
            if haveTestData then
                print(string.format('Creating new test data : %d samples of size  %s (%.1f MB)',  
                        nTestSamples, toList( feat_extr_outputSize, nil, ' x '), sizeOfNewTestInputs_MB ))
                local newInputSize = torch.concat(nTestSamples, feat_extr_outputSize)
                
                local newTestInputs = torch.Tensor(newInputSize):typeAs(sample_outputFromFeat)
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
                testData_use  = {[input_field] = testInputs, [output_field]=testingOutputs, 
                                nInputs = feat_extr_nOutputs, nOutputs = nOutputs, nPositions = testData.nPositions}
            end
        end
        
     
    end
    
    --Model = model_struct
    TrainData = trainData_use
    MTrain = model_toTrain
    
    -- if freezeFeatures, model_forTest will be updated to be just the classifier
    model_struct_forTest = {model = model_forTest, criterion=criterion, modelType = 'model_struct', 
                            outputTargetField = model_struct.outputTargetField,
                            expandedTableModel = model_struct.expandedTableModel,
                            tableModel = model_struct.tableModel,
                            modelToUse = model_struct.modelToUse, }

    
    --TrainData_use = trainData_use
    --print('GetParameters')
    local parameters, gradParameters = model_toTrain:getParameters()
    modelP = parameters
        
    --print('Shuffle')
    
    local trainingIdxs = torch.range(1, nTrainingSamples)
    if shuffleTrainingSamples then        
        torch.manualSeed(123)
        trainingIdxs = torch.randperm(nTrainingSamples)
    end
    
    local nInputFeats 
    if torch.isTensor( trainInputs[1] ) then
        nInputFeats = torch.nElements( trainInputs[1] )
    end
    if freezeFeatures and not usePreExtractedFeatures then
        nInputFeats = feat_extr_nOutputs
    end

    count = 0;
    
    local checkForNan = false
    
    local _idx_ = 0
    
    -- define closure to calculate cost/gradient of cost function 
    local feval = function(param_new)
                
        if parameters ~= param_new then
            parameters:copy(param_new)
        end
        gradParameters:zero()
        G = gradParameters
        
        local loss = 0;
        local nThisBatch = batchSize
        --GroupBatch = groupBatch
        if not groupBatch then
            GroupBatch = false  
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
                local curTrainIdx = trainingIdxs[_idx_]
                if useFunctionToGetTrainingIdx then
                    curTrainIdx = getTrainingIdx()
                    CurIdx = curTrainIdx
                end
                   
                T = trainInputs
                input = trainInputs[curTrainIdx]    
                --input_copy = input:clone()
                target = trainingOutputs[curTrainIdx]  
                --target_copy = target:clone()
                               
                if freezeFeatures and not usePreExtractedFeatures then   -- extract features now to pass the the (trainable) classifier                    
                    input = feat_extractor:forward(input)
                    count = count + 1
                end
 
                if reprocessInputs then
                    -- e.g. shiftScaleDownImage:   for patch-classifier training, can zoom in to a different subset (90x90) of the original patch (100x100)
                    -- e.g. replicateImageAtScales:for multi-scale training, replicate image at different scales before passing it to the table model
                    input = reprocessInputs(input)
                end
                
                TrOp = trainingOpts
                if reprocessInputsAndTargets then  
                    -- e.g. shiftImageAndPos : for heatmap backend regression: scale(zoom in/out)+shift both the input heatmap and target node locations 
                    input, target = reprocessInputsAndTargets(input, target, curTrainIdx) 
                end
 
                if resizeInputToVector then           -- even if model already reshapes input, 
                    input = input:resize(nInputFeats)  -- model:backward needs vector for simple networks
                end
                    
                Model_toTrain = model_toTrain
                Crit = criterion

                local output = model_toTrain:forward(input)
                local nOutputsThisTime = output:numel()                
                
                Output = output
                if not network_sampleOutput then
                    network_sampleOutput = output
                end
                if output_field == 'outputMatrix' and false then
                    assert(output:dim() == 1)
                    assert(nOutputsThisTime == nOutputs)
                    
                    if output:dim() > 1 then
                        output = output:resize(nOutputs)
                    end
                    
                end
    
                --error('!')
                --input, target = TrOp.reprocessInputsAndTargets(input_copy, target_copy)                                                     
                --mattorch.save('test5.mat', {x=input_copy:double(), x1 = input[1]:double(), x2 = input[2]:double(), t = target:double(), v=target_copy:double(), tmod=Tmod:double()})
    
                
    
                --allOutputs = allOutputs or {}
                --table.insert(allOutputs, output:clone())
                showDecreasingOutput = false
                if showDecreasingOutput then 
                    minOutput = minOutput or 1e10
                    output_sum = output:sum()
                    if output_sum < minOutput then
                        minOutput = output_sum 
                        progressBar.printf('\nidx = %d. smallest output is now %.5f\n', _idx_, minOutput)
                    end
                    maxOutput = maxOutput or 0
                    if output_sum > maxOutput then
                        maxOutput = output_sum 
                        progressBar.printf('\nidx = %d. largest output is now         %.5f\n', _idx_, maxOutput)
                    end
                    
                end
    
                -- we can adjust the heatmap targets so the bumps get successively narrower over the course of training
                if targetsDependOnEpoch then
                    --tic() 
                    target = targetFunction(curTrainIdx, epoch_id)
                    --print( toc() )
                    
                    trainingOutputs[curTrainIdx] = target
                end
                --error('!')
                
                if torch.typename(criterion) == 'nn.WeightedMSECriterion'  then
                
                    if criterion.MSEWeightsDependOnTarget then
                        -- adjust the weights of the MSE criterion for this specific target (to emphasize the bumps more than the flat parts)
                        
                        if targetsDependOnEpoch and weightsFunctionEpoch ~= epoch_id then
                            -- the MSEWeightsFunction depends on the statisics of the targets. if these have changed, have to adjust the function
                            weightsFunctionEpoch = epoch_id 
                            
                            local a,b = getWeightScaleParams(trainingOutputs)
                            
                            --local weights = t*a + b
                            --Weights = weights
                            cprintf.Cyan('\nUpdating weighted MSE criterion, epoch  %d : a = %.2f, b = %.2f\n', epoch_id, a, b)
                            
                            criterion.MSEWeightsFunction = function(t)
                                return t * a + b
                            end
                            
                        end
                        
                        if criterion.MSEUpdateWeightsFunction then
                            criterion.MSEUpdateWeightsFunction(epoch_id, _idx_)
                        end
                        criterion.weight = criterion.MSEWeightsFunction(target)
                        --curTarget = 
                    elseif criterion.MSEWeightsDependOnTargetAndOutput then
                        criterion.weight = criterion.MSEWeightsFunction(target, output)
                        
                        iii = iii or 1
                        iii = iii + 1
                
                        
                        
                    end
                        
                        
                end
                  
                Criterion_use = criterion
            
            
                if trainingOpts.useOutputAndTarget then
                    trainingOpts.useOutputAndTarget(output, target, curTrainIdx, criterion)
                end
                    
    
            
            --[[
                local extraLoss 
                if torch.typename(criterion, 'nn.ClassNLLCriterion') and output:nDimension() == 3 then
                    extraLoss = 0
                    
                    output_mtx = Output:permute(2,3,1):reshape(sz[2]*sz[3], sz[1]):size()
                
                else
                --]]
                
                local extraLoss = criterion:forward(output, target)
                ExtraLoss = extraLoss
                gradBack = criterion:backward(output, target)
                
                model_toTrain:backward(input, gradBack)
                
                --if _idx_ < 10 then
                  --  io.write(string.format('%d : output : %s \n       target: %s\n.             extraLoss: %.4f\n\n', 
                    --            _idx_, tostring_inline(output, '%.3f'), tostring_inline(target, '%.3f'), extraLoss))
                -- end
                loss = loss + extraLoss
                if _idx_ <= 10 then
                    --cprintf.Red('Loss = %.10f. Output[1] = %.10f. W[1] = %.10f. P = %.10f\n', 

                      --  ExtraLoss, Output:view(-1)[1], firstWeightVal( findModuleOfType(model_struct.model, 'linear', 1) ), modelP:sum()  );
                    
                    --progressBar.printf('.')
                end
                --model_toTrain:backward(input, criterion:backward(output, target))
               
                if trainingClassifier then
                    trainConfusionMtx:add(output:view(-1),target)          
                end
                
                --error('!')
                
               
                showOutputAverages = false
                
                if showOutputAverages then
                    outputAverageN = 200
                    
                    if not allOutputSums then
                        allOutputCorr = torch.Tensor(outputAverageN)
                        allOutputSums = torch.Tensor(outputAverageN)
                        allTargetSums = torch.Tensor(outputAverageN)
                    end
                    
                    --progressBar.printf('[%d]', _idx_)
                
                    cur_idx =  ( (_idx_-1) % outputAverageN)+1
                    if string.find(  torch.typename(criterion), 'MSECriterion') then
                        --outputAverageN = math.ceil( nTrainingSamples/10 )
                        k = _idx_
                        allOutputSums[ cur_idx] = output:sum()
                        allTargetSums[ cur_idx] = target:sum()
                      if _idx_ % outputAverageN == 0 then
                            progressBar.printf('\n- i = %d.    output_av =  %.10f.   target_av = %.10f --\n', 
                                _idx_, allOutputSums:mean(), allTargetSums:mean())
                      end
                        
                    elseif torch.typename(criterion) == 'nn.ClassNLLCriterion' then
                        
                        mx, idx_select = output:max(1)
                        isCorr = iff(idx_select[1] == target, 1, 0)
                        
                        allOutputCorr[ cur_idx] = isCorr
                        allOutputSums[ cur_idx] = extraLoss
                        
                        
                      --if _idx_ % outputAverageN == 0 then
                            --progressBar.printf('\n- i = %d.    loss_av =  %.10f.   pct_correct_av = %.10f --\n', 
                            --    _idx_, allOutputSums:mean(), allOutputCorr:mean()*100)
                        --error('!')
                      --end
                        
                    end
                        
                end

                
                
                
                
            end
            
        elseif groupBatch then
            GroupBatch = true
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
            
            progressBar.step(_idx_)
            local sampleIdxs = trainingIdxs[{batchIdxRange}]
            TrainInputs = trainInputs
            --inputs = permuteBatchDimTo4(trainInputs, batchIndices)
            local inputs = selectInputBatch(trainInputs, sampleIdxs)
            Inputs = inputs
            Feat_extractor = feat_extractor
            if freezeFeatures and not usePreExtractedFeatures then
                inputs = feat_extractor:forward(inputs)
            end
            local targets = selectInputBatch(trainingOutputs, sampleIdxs)

            --inputs = trainInputs[{{batchIndices}}]
            
            if resizeInputToVector then                          
                inputs = inputs:resize(nThisBatch, nInputFeats)  -- resize input appropriately
            end
            Model_toTrain = model_toTrain
            InputsNow = inputs
                        
            local outputs = model_toTrain:forward(inputs)
            assert(outputs:size(1) == nThisBatch)
            assert(outputs:size(2) == nOutputs)
            
            --local input = trainInputs[trainingIdxs[_idx_]]                                    
            --local target = trainingOutputs[trainingIdxs[_idx_]]  

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
                if trainingClassifier then
                    trainConfusionMtx:add(outputs[i],targets[i])          
                end

            end
            --model:backward(inputs, df_do:cuda())
                                     
            model_toTrain:backward(inputs, df_do)
            _idx_ = _idx_ + nThisBatch-1
            
            progressBar.step(_idx_)
        
        end
    
       
        if nThisBatch > 1 then
            --loss = loss / nThisBatch  (loss is added up, not averaged)
            gradParameters = gradParameters:div(nThisBatch)
        end
                
        return loss, gradParameters
       
    end

    
    -- repeat train/test loop until stop criterion is satisfied
    
    local logger
    if trainingOpts.SAVE_TRAINING then
        
        paths.createFolder(trainingDir)
        
        logger = optim.Logger2(csvLogFile, logger_open_mode)
    end

    
    local do_SGD_Loop = function()
        model_toTrain:training()
        local loss = 0
            
        for j = 1, nTrainingSamples/batchSize do
                    --cprintf.Cyan('Got HERE!')
            local _,fs = optim.sgd_auto(feval,parameters,trainingLogger.sgd_config)
            --allLoss[j] = fs[1]
                       
            loss = loss + fs[1]
        end
        
        return loss --/ nTrainingSamples
    end
    
    
    if continueTraining then
        print('Training => ' .. torchLogFile)
    end
    --print(trainingLogger.params)
                
    local current_train_loss, prev_train_loss, train_loss_change_pct = 0,0,0
    local current_test_loss, prev_test_loss, test_loss_change_pct = 0,0,0
    local current_test2_loss, prev_test2_loss, test2_loss_change_pct = 0,0,0
    local current_train_loss_extraCrit, current_test_loss_extraCrit = 0,0
    local curr_trainErr_pct, prev_trainErr_pct, trainErr_pct_change_pct = 100, 100, 100
    local curr_testErr_pct, prev_testErr_pct, testErr_pct_change_pct = 100, 100, 100
    local curr_test2Err_pct, prev_test2Err_pct, test2Err_pct_change_pct = 100, 100, 100
    local fs

       
    local checkErrBeforeStart = false and (trainingLogger.nEpochs == 0)
    if checkErrBeforeStart then
        curr_trainErr_pct, _, current_train_loss = testModel(
            model_struct_forTest, trainData_use, {getLoss = true, batchSize = batchSize, 
                                                reprocessInputs = reprocessInputs, reprocessInputsAndTargets = reprocessInputsAndTargets})        
        
        prev_train_loss = current_train_loss
        
        curr_testErr_pct, _, current_test_loss = testModel(
            model_struct_forTest, testData_use, {getLoss = true, batchSize = batchSize, 
                                                reprocessInputs = reprocessInputs, reprocessInputsAndTargets = reprocessInputsAndTargets})        
        prev_test_loss = current_test_loss
        
        io.write(string.format('   Quick check: Train Loss = %.4f, Test Loss = %.4f. ', current_train_loss, current_test_loss))
            
        if showErr then
            io.write(string.format('TrainErr = %.3f. TestErr = %.3f', curr_trainErr_pct, curr_testErr_pct))
        end
        io.write('\n')
            
            
        prev_train_loss   = current_train_loss
        prev_test_loss    = current_test_loss
        prev_trainErr_pct = curr_trainErr_pct
        prev_testErr_pct  = curr_testErr_pct
    end
    
    
    
    while continueTraining do
        if criterion_firstEpoch then
            --nEpochsDoFirstOne = 1
            
            if epoch_id <= nEpochsFirstCriterion then
                criterion = criterion_firstEpoch
                cprintf.cyan('Epoch #%d : Using special criterion : %s\n', epoch_id, tostring(criterion))
            elseif epoch_id > nEpochsFirstCriterion  then
                criterion = model_struct.criterion
                cprintf.cyan('Epoch #%d : Using this criterion for the remaining epochs : %s\n', epoch_id, tostring(criterion))
            end
        end
        
        cprintf.Green('Starting Epoch %d with %s: ', epoch_id, trainingAlgorithm)
        io.flush()
        
        local startTime = os.time()

        if trainingAlgorithm == 'SGD' then

            batchSize = trainingOpts.BATCH_SIZE or 1
            BatchSize = batchSize
            progressBar.init(nTrainingSamples, progressBar_nStars) --- progressbar for SGD
        
            if trainingClassifier then
                trainConfusionMtx:zero()
            end
                            
            local startTime = os.time()
            current_train_loss = do_SGD_Loop()
            progressBar.done()
            
            current_train_loss = current_train_loss / nTrainingSamples
           
            if trainingClassifier then
                trainConfusionMtx:updateValids()
                curr_trainErr_pct = (1-trainConfusionMtx.totalValid)*100   --trainError = testModel(model, testData)        
            end
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
            current_train_loss = fs[1]
    
            curr_trainErr_pct = testModel(model_struct_forTest, trainData_use, {verbose = showTrainTestTime, test_indiv_pos = trainOnIndividualPositions, batchSize = batchSize, reprocessInputs = reprocessInputs, reprocessInputsAndTargets = reprocessInputsAndTargets}) -- L-BFGS does multiple loops
                
        end
       
        local timeForThisEpoch = os.time() - startTime
        if showTrainTestTime and (trainingAlgorithm == 'SGD') then
            local timeEachSample_sec = (timeForThisEpoch / nTrainingSamples)
            print('==> time to train each sample = ' .. timeEachSample_sec *1000 .. 'ms')
        end


      
        
     
        
        
            local t_elapsed_sec = 0
            local trainErr_pct_change_pct = (curr_trainErr_pct - prev_trainErr_pct)/prev_trainErr_pct*100 
            local testErr_pct_change_pct  = (curr_testErr_pct - prev_testErr_pct)/prev_testErr_pct*100 
            prev_trainErr_pct = curr_trainErr_pct
            
            if haveTestData then
                cprintf.blue('     > Evaluating performance on testing set      : ')

                curr_testErr_pct, _, current_test_loss, t_elapsed_sec = 
                    testModel(model_struct_forTest, testData_use, {getLoss = true, verbose = showTrainTestTime, 
                                test_indiv_pos = trainOnIndividualPositions, batchSize = batchSize, 
                                reprocessInputs = reprocessInputs, reprocessInputsAndTargets = reprocessInputsAndTargets})        
            else
                curr_testErr_pct = 0
            end
        
            if haveSecondTestData and test2Data.outputMatrix then
                curr_test2Err_pct, _, current_test2_loss, t_elapsed_sec = 
                    testModel(model_struct_forTest, test2Data, {getLoss = true, verbose = showTrainTestTime, 
                                test_indiv_pos = trainOnIndividualPositions, batchSize = batchSize, 
                                reprocessInputs = reprocessInputs, reprocessInputsAndTargets = reprocessInputsAndTargets})        
            else
                curr_test2Err_pct = 0
            end
        
        
        if extraCriterion then
            model_struct_forTest.criterion = extraCriterion
            
            cprintf.blue('     > Evaluating extra Criterion on training set : ')
            _, _, current_train_loss_extraCrit = testModel(model_struct_forTest, trainData_use, 
                {getLoss = true, verbose = showTrainTestTime, test_indiv_pos = trainOnIndividualPositions, 
                        batchSize = batchSize, reprocessInputs = reprocessInputs})        

            cprintf.blue('     > Evaluating extra Criterion on testing set  : ')
            _, _, current_test_loss_extraCrit =  testModel(model_struct_forTest, testData_use, 
                {getLoss = true, verbose = showTrainTestTime, test_indiv_pos = trainOnIndividualPositions, 
                    batchSize = batchSize, reprocessInputs = reprocessInputs})        
                
            model_struct_forTest.criterion = origCriterion
            
        end
        
        
        
        train_loss_change_pct = (current_train_loss - prev_train_loss)/prev_train_loss * 100
        prev_train_loss = current_train_loss;

        test_loss_change_pct = (current_test_loss - prev_test_loss)/prev_test_loss * 100
        prev_test_loss = current_test_loss;
        
        test2_loss_change_pct = (current_test2_loss - prev_test2_loss)/prev_test2_loss * 100
        prev_test2_loss = current_test2_loss;
        
        --io.write('[print]')
        
        --io.write('[add\n');
        printf(' epoch id == %d\n', epoch_id)
        trainingLogger:add(epoch_id, model_struct, current_train_loss, curr_trainErr_pct, curr_testErr_pct, timeForThisEpoch)
        --io.write(']')
        
        
        --io.write('[Cont')
        continueTraining, trainingAlgorithm, reasonToStopTraining = trainingLogger:continue()
        --io.write(']')
        
        
        local checkErrAfterEachEpoch = false
        if checkErrAfterEachEpoch then
            cprintf.blue('     > Evaluating performance on Training set      : ')

            local curr_trainErr_pct_after, _, current_train_loss_after = 
                testModel(model_struct_forTest, trainData_use, {getLoss = true, batchSize = batchSize, 
                                                                reprocessInputs = reprocessInputs, reprocessInputsAndTargets = reprocessInputsAndTargets})

            cprintf.blue('     > Evaluating performance on Testing set       : ')
            local curr_testErr_pct_after, _, current_test_loss_after = 
                testModel(model_struct_forTest, testData_use, {getLoss = true, batchSize = batchSize, 
                                                               reprocessInputs = reprocessInputs, reprocessInputsAndTargets = reprocessInputsAndTargets})
                
            io.write(string.format('   Quick check: TrainLoss = %.4f. TestLoss = %.4f. ', 
                                    current_train_loss_after, current_test_loss_after))        
            if showErr then
                io.write(string.format('TrainErr = %.3f. TestErr = %.3f', curr_trainErr_pct_after, curr_testErr_pct_after))        
            end
            io.write('\n')
        end
        

        
        if trainingOpts.SAVE_TRAINING then
            tic()
            io.write(string.format('[Saving...' ))
            --cleanModel(model_struct)
            --convertModelToFloat(model_struct)
            
            trainingLogger:saveToFile()
            --cprintf.Red('Not Saving\n');
            
            if trainOnGPU then
              --  model_toTrain:cuda()
            end
            
            local t_elapsed = toc()
            io.write(string.format('done:%s]', sec2hms(t_elapsed)))
            
            logger:add{['Epoch'] = epoch_id}        
            logger:add{['Train Cost'] = current_train_loss}
            logger:add{['Test  Cost'] = current_test_loss}
            if haveSecondTestData then
                logger:add{['Test2 Cost'] = current_test2_loss}
            end
            logger:add{['Train Err'] = curr_trainErr_pct}
            logger:add{['Test Err'] = curr_testErr_pct}
            if haveSecondTestData then
                logger:add{['Test2 Err'] = curr_test2Err_pct}
            end
            
            if extraCriterion then
                logger:add{['Train Cost 2'] = current_train_loss_extraCrit}
                logger:add{['Test  Cost 2'] = current_test_loss_extraCrit}
            end
            logger:add{['Time'] = timeForThisEpoch}
                
            logger:println()
            
        end
        model_struct.trainingDate = os.time()

        local timeForThisEpoch_tot = os.time() - startTime
        local current_train_loss_per_pixel_str, current_test_loss_per_pixel_str = '', ''
        if torch.typename(criterion) == 'nn.MSECriterion' and criterion.sizeAverage == true  then
            current_train_loss_per_pixel_str = string.format(' [%.2f pp]', current_train_loss * network_sampleOutput:numel())
            current_test_loss_per_pixel_str = string.format(' [%.2f pp]', current_test_loss * network_sampleOutput:numel())
        end
        
        io.write(string.format('   after epoch %d: TrainCost = %.4f%s (%+.2f%%). TestCost = %.4f%s (%+.2f%%)', 
                epoch_id, current_train_loss, current_train_loss_per_pixel_str, train_loss_change_pct, 
                              current_test_loss, current_test_loss_per_pixel_str, test_loss_change_pct ))

        if showErr then
            io.write(string.format('TrainErr = %.3f (%+.2f%%). TestErr = %.3f  (%+.2f%%)', 
                        curr_trainErr_pct, trainErr_pct_change_pct, curr_testErr_pct, testErr_pct_change_pct ))
        end
        io.write(string.format(' [took %s]\n', sec2hms(timeForThisEpoch_tot)))


        if haveSecondTestData then
            io.write(string.format('    [Second test set: Cost = %.4f (%+.2f%%). Test : %.4f]\n', 
                        current_test2_loss, test2_loss_change_pct,  curr_test2Err_pct))
           
        end


        if extraCriterion then
            io.write(string.format('    [Extra Criterion: Train: %.4f. Test : %.4f]\n', current_train_loss_extraCrit, current_test_loss_extraCrit))
            
           convertCostToPixels = true and torch.typename(extraCriterion) == 'nn.mycri'
            if convertCostToPixels then
                local imageSize = {trainData_use.inputMatrix:size(3), trainData_use.inputMatrix:size(4)}
                local train_meanDist_image, train_meanDist_pix = cost2dist(current_train_loss_extraCrit, imageSize)
                local test_meanDist_image,  test_meanDist_pix  = cost2dist(current_test_loss_extraCrit, imageSize)
                cprintf.Red('      Train Error in pixels : %.4f of image / %.2f pixels\n', train_meanDist_image, train_meanDist_pix)
                cprintf.Red('      Test  Error in pixels : %.4f of image / %.2f pixels\n', test_meanDist_image, test_meanDist_pix)
            end
            
        end

        io.write('\n')
        
        if trainingOpts.saveHeatmapsAfterEachEpoch then
            reprocessInputsForTraining = false
            
            for i = 1,2 do
                fn_base = trainingOpts.outputHeatmapsFiles[i]
                if i == 1 then
                    nm = 'training' 
                    dataset = trainData
                elseif i == 2 then
                    nm = 'testing'
                    dataset = testData
                end
                fn = string.gsub(fn_base, '.mat', '__' .. epoch_id .. '.mat')
                
                cprintf.Cyan('Generating %s heatmaps --> %s\n', nm, fn);
            
                outputHeatmaps_tmp = generateOutputHeatmapsForBatch(model_struct, dataset.inputMatrix, trainingOpts.reprocessInputsAndTargets)
                
                S_heatmaps = {outputHeatmaps = heatmap2int(outputHeatmaps_tmp ), targetHeatmaps = heatmap2int (dataset.outputTargets), 
                                outputMatrix   = dataset.outputMatrix:float() }
        
                paths.verifyFolderExists(fn, true) -- if doesn't exist, create it now                        
                matio.save(fn, S_heatmaps)
            end
                
            reprocessInputsForTraining = true;
            
        end
        
        
        
        do
            --error('!')
        end
        if not continueTraining then
            io.write(string.format('Stopped training : %s\n', reasonToStopTraining))
        end
        
        -- check for signal to error out & restart!
        checkForErrorSignal()
        
         
         
        epoch_id = epoch_id + 1;
        
   
    end

    return model_struct   -- important to return this at the end -- in case have loaded saved file and are replacing original one

    
end




testModel = function(model_struct, testData, opt)
    MSS =model_struct
    TSS = testData
    assert(#testData == 0)
    opt = opt or {}
    local verbose = opt.verbose or false
    local getLoss = opt.getLoss or false
    local multipleLabels = opt.multipleLabels or false
    local test_indiv_pos = opt.test_indiv_pos or false
    nOutputs = testData.nClasses or testData.nOutputs
    local returnPctCorrect = opt.returnPctCorrect or false 
    if torch.isTensor(nOutputs) then
        nOutputs = nOutputs[1][1]
    end

    tic()
    local model
    if (model_struct.modelType ~= nil) then
        --print('test mode 1')
        model = model_struct.modelToUse or model_struct.model
    else
        --print('test mode 2')
        model = model_struct
    end
    
    model:evaluate()
    
    local groupBatch = areAnySubModulesOfType(model, 'nn.SpatialConvolutionCUDA') 
    local batchSize = 1
    if (opt.batchSize and groupBatch) then
        batchSize = opt.batchSize
    end


    local trainingClassifier = torch.isClassifierCriterion(torch.typename(model_struct.criterion))
    
    local output_field 
    if trainingClassifier then
        output_field = 'labels'
        --output_field = 'outputMatrix'
        if test_indiv_pos then
            output_field = 'labels_indiv_pos'
            nOutputs = nOutputs * testData.nPositions
        end
    elseif model_struct.outputTargetField then
        output_field = model_struct.outputTargetField
    else
        output_field = 'outputMatrix'
    end
    OF = output_field
    
    
    local input_field = model_struct.inputDataField or 'inputMatrix'

    local useConfusionMatrix = true and trainingClassifier and not multipleLabels
    
	local testInputs = testData[input_field]

	local outputs = testData[output_field]
    local haveSecondLabels = testData.labels_distract ~= nil   -- for testing with multiple labels
    local haveThirdLabels  = testData.labels_distract2 ~= nil

    local testConfusionMtx
    if useConfusionMatrix then
        testConfusionMtx = optim.ConfusionMatrix(nOutputs)
        testConfusionMtx:zero()
    end

    
    --testInputs_copy
	i = 0;
    if verbose then
        tic()
        --print('==> testing ')
    end
 
        
    local size1 = function(x)
        if torch.isTensor(x) then 
            return x:size(1)
        elseif type(x) == 'table' then 
            return #x
        end
    end
 
 
	local nTestSamples = size1(testInputs)
    local loss = 0
    local nCorrect = 0
    local criterion
    local target, idx_max
    if getLoss then
        if model_struct.criterion then
            criterion = model_struct.criterion
        else
            error('No Criterion in model_struct')
        end
    end
    CHECKED_MSE = false
    TestInputs = testInputs
    Outputs = outputs
    M = model
    local MSEWeightsDependOnTarget =  (torch.typename(criterion) == 'nn.WeightedMSECriterion') and criterion.MSEWeightsDependOnTarget

    if MSEWeightsDependOnTarget then
        cprintf.yellow('MSE depends on target\n')
    end
    
    local nBatches = math.ceil(nTestSamples / batchSize)
    local pred
    
    local doProgressBar = opt.doProgressBar or true
    if doProgressBar then
        progressBar.init(nBatches, 20)
    end
	for batch_i = 1, nBatches do

        
        local idxs = { (batch_i-1) * batchSize +1,  math.min(batch_i * batchSize, nTestSamples) } 
        local nThisBatch = idxs[2] - idxs[1] + 1
        
        Idxs = idxs
        local preds, targets
        
        if groupBatch then        
            preds = model:forward(testInputs[{ idxs }])
            --Preds = M:forward(TestInputs[{ idxs }])
            targets = outputs[{idxs}] 
        else
            local input = testInputs[{ idxs[1] }]
            targets = outputs[{idxs[1]}] 
            
            if  opt.reprocessInputs then
                input = opt.reprocessInputs(input)
            end
            if  opt.reprocessInputsAndTargets then
                input, targets = opt.reprocessInputsAndTargets(input, targets)
            end
            II = input
            TT = targets
            MM = model
            
            preds = model:forward( input )
        end
        
        P = preds
        T = targets
        
      
        
        for samp_i = 1, nThisBatch do
            NB = nBatches
            NN = nThisBatch
            Si = samp_i
        
            if preds:dim() == 1  or not groupBatch then
            --if groupBatch and nThisBatch > 1 then
                pred = preds
            else 
                pred = preds[samp_i]
            end
        
            
            if trainingClassifier then
        
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
                    _,idx_max = torch.max(pred:resize(nOutputs), 1)
                    --local _,idx_max = torch.max(pred, 1)

                    if idx_max[1] == testData.labels[idx_sample] then
                        nCorrect = nCorrect + 1
                    elseif multipleLabels and
                        (haveSecondLabels and (idx_max[1] == testData.labels_distract[idx_sample])) or
                        (haveThirdLabels  and (idx_max[1] == testData.labels_distract2[idx_sample])) then
                            nCorrect = nCorrect + 1
                    end
                end        
            else
                target = targets
            end
        
           
            if getLoss then
                i = i + 1
                --io.write('^');
                P1 = pred
                T1 = target
                C = criterion
                lossBefore = loss
                
                local criterion_use = criterion

                if torch.typename(criterion_use) == 'nn.WeightedMSECriterion' and criterion_use.MSEWeightsDependOnTarget then                    
                    criterion_use.weight = criterion_use.MSEWeightsFunction(target)
                end
                
                local extraLoss = criterion_use:forward(pred, target)
                if loss ~= loss then
                    error('got a nan')
                end
                
                loss = loss + extraLoss
                
                
            end
       
            
            
        end
        
        if doProgressBar then
            progressBar.step()
        end
        
	end
        
    if doProgressBar then
        progressBar.done(opt.removeProgressBarWhenDone)
    end
    
 
    local totalTime
    if verbose then
        totalTime = toc()
    end

    local t_elapsed = toc()
    
    if getLoss then
        loss = loss / nTestSamples
    end

    if not trainingClassifier then
        return 0, 0, loss
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
        testErr_pct_eachClass = torch.Tensor(nOutputs):zero()
    
    end

  
    if verbose then
        local timeEachSample_sec = (totalTime / nTestSamples)
        io.write(string.format('\n [test]: %.2f ==> Total time: %.3f sec. Time for each sample = %.3f ms. \n', 
                testErr_pct_total, totalTime, timeEachSample_sec *1000))
        --print(testConfusionMtx)
    end
    
    if returnPctCorrect then
        local testCorrect_pct     = 100 - testErr_pct_total            -- a number
        testCorrect_pct_eachClass =      -testErr_pct_eachClass + 100  -- a tensor (so need to put first to add a number)
        return testCorrect_pct, testCorrect_pct_eachClass, loss, t_elapsed
    else

        return testErr_pct_total, testErr_pct_eachClass, loss, t_elapsed
    end

end


--]]


--[[
getTimeStr = function()
    return os.date("%H_%M_%S")
end
--]]

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


cost2dist = function(cost, imageSize)
    local meanDist_image = math.sqrt(cost/2)
    local meanDist_pix   = meanDist_image * (  (imageSize[1] + imageSize[2])/2  )
    return meanDist_image, meanDist_pix
end


checkForErrorSignal = function()
    if lettersData_dir and timeStarted then
        local stop_fname = lettersData_dir .. 'stop'
        if paths.filep(stop_fname) and paths.filedate(stop_fname) > timeStarted then
            
            io.write(string.format('Stop file signal date : (%s) \nis later than we started (%s)', 
                    os.date('%x %X', paths.filedate(stop_fname)), os.date('%x %X', timeStarted) ) )
                    
            local nSecSleep = (torch.random() % 5) + 15
            printf('Pausing for %d seconds ...', nSecSleep);
            io.flush()
            sys.sleep(nSecSleep)
            print('Aborting ...!')
            os.exit()
        end
        
    end
        
end


