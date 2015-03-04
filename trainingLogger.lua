

    if not TrainingLogger then
        require 'optim'
        require 'sys'
        TrainingLogger = torch.class('optim.TrainingLogger')
    end
    
    --print('Reloading ... ')
    function TrainingLogger:__init(newParams)    
        self.params = {}
        self.params.COST_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
        self.params.TRAIN_ERR_CHANGE_FRAC_THRESH = 0.001  -- = 0.1%
        self.params.TEST_ERR_NEPOCHS_STOP_AFTER_MIN = 10
        self.params.MIN_EPOCHS = 10
        self.params.MAX_EPOCHS = 500
        self.params.EXTRA_EPOCHS = 5 -- after satisfies stopping criteria -- do another few epochs
        self.params.SWITCH_TO_LBFGS_AT_END = false
        self.params.LBFGS_USE_REDUCED_SET_FIRST = false
        self.params.REQUIRE_COST_MINIMUM = false -- require cost function to reach local 
        self.params.REQUIRE_TRAINING_ERR_MINIMUM = true
        self.params.SAVE_TRAINING = true
                                  -- minimum (vs use either cost/train error as criterion)
        
        self.nEpochs = 0
        self.nExtraEpochs = 0
        self.filename = {}
        self.satisfiedStopCriterion = false
        --self.prevTrainingState = 'SGD'
        self.trainingState = 'SGD' -- possible states: 'SGD', 'L-BFGS', 'L-BFGS-reduced', 'DONE'
        self.reason = ''
        
        self.cost = {}
        self.trainErr = {}
        self.testErr = {}
        self.dCost = {}
        self.dTrainErr = {}
        self.dTestErr = {}
        
        self.epochTrainingTime = {}
        self.lastUpdated = {}
        
        if newParams ~= nil then
            self:setOptions(newParams)
        end
    end

    function TrainingLogger:setOptions(newParams)
        local otherParams = {freezeFeatures = 1, LBFGS_REDUCED_FRAC = 1}
        local errorIfUnknownParam = false
        if newParams then
            for k,v in pairs(newParams) do
                if self.params[k] == nil and not (otherParams[k]) and errorIfUnknownParam then
                    error('Tried to set unknown parameter: ' .. k ) 
                end
                --io.write(string.format(" - Training option : %s := %s\n", tostring(k), tostring(v)))
                self.params[k] = v
            end
        end
        --Self = self
        
    end

    function TrainingLogger:epochOfMinTestError()
        local epochMin = 1
        local minTestErr = 100
        
        for idx,testErr_i in ipairs(self.testErr) do
            if testErr_i < minTestErr then
                minTestErr = testErr_i
                epochMin = idx;
            end
        end
        
        return epochMin, minTestErr
    end

    function TrainingLogger:add(epoch, model_struct, cost, trainErr, testErr, epochTrainingTime)
    
        if model_struct then
            self.model_struct = model_struct
            self.epochOfModel = epoch
        end
        
        if cost or trainErr or testErr or epochTrainingTime then
            self.nEpochs = epoch
            --print('epoch = ' .. epoch)
            
            if cost then
                self.cost[epoch] = cost
                                
                if epoch == 1 then
                    self.dCost[epoch] = 0
                elseif epoch > 1 then
                    self.dCost[epoch] = math.abs( self.cost[epoch] - self.cost[epoch-1] )
                end
                
            end
            if trainErr then
                self.trainErr[epoch] = trainErr
                
                if epoch == 1 then
                    self.dTrainErr[epoch] = 0
                elseif epoch > 1 then
                    self.dTrainErr[epoch] = math.abs( self.trainErr[epoch] - self.trainErr[epoch-1] )
                end
                
            end
            if testErr then
                self.testErr[epoch] = testErr
            end
            if epochTrainingTime then
                if not self.epochTrainingTime then
                    self.epochTrainingTime = {}
                end
                self.epochTrainingTime[epoch] = epochTrainingTime
            end
            
            self.lastUpdated = os.time()
            
            self.forcingContinue = nil -- after have added one epoch, remove forcingContinue flag
        else
            error('No input')
        end
    
    end
 
    function TrainingLogger:currentLoss()
        return self.cost[self.nEpochs]
    end
    function TrainingLogger:currentTrainErr()
        return self.trainErr[self.nEpochs]
    end
    function TrainingLogger:currentTestErr()
        return self.testErr[self.nEpochs]
    end
        
    function TrainingLogger:lastUpdatedTime()
        self.lastUpdated = self.lastUpdated or 0 
        return self.lastUpdated
    end
        
    function TrainingLogger:totalTime()
        if self.epochTrainingTime then
            local tot = 0
            for _,t in ipairs(self.epochTrainingTime) do
                tot = tot + t
            end
            return tot
        else
            return 0
        end
    end
        
    function TrainingLogger:continue()
        p = self.params
    
      -- wait until (1) have very little change in cost function (<0.1% change over nAverage epochs). 
      -- but, also have a minimum of MIN_EPOCHS. and if gradient started off small, wait until got at least twice max gradient of beginning, before allowing to settle down.
      -- is if  gradient that is greater than initial gradient 
      -- keep 

        
        local nEpochs = self.nEpochs
                    
        if nEpochs > 1  then -- and self.trainingState == 'SGD' then
            self.dCost_frac = self.dCost_frac or {}
            self.dTrainErr_frac = self.dTrainErr_frac or {}
            
            self.dCost_frac[nEpochs] = self.dCost[nEpochs] / (self.cost[nEpochs-1] + 1e-5)
            self.dTrainErr_frac[nEpochs] = self.dTrainErr[nEpochs] / (self.trainErr[nEpochs-1] + 1e-5)
           

            --print(self.dTrainErr_frac[nEpochs] .. ' = ' ..  self.dTrainErr[nEpochs] .. ' / ' .. self.trainErr[nEpochs-1])

            --nAverage = 1
            --iStart = math.max(1, nEpochs-nAverage)

            --for i = iStart, nEpochs do
            --    self.dcost_frac_rav[nEpochs] = self.dcost_frac_rav[nEpochs] +  self.dcost_frac[i]
            --end
            --self.dcost_frac_rav[nEpochs] = self.dcost_frac_rav[nEpochs] / (nEpochs - iStart + 1)
            
            local cost_below_change_threshold = (self.dCost_frac[nEpochs] < p.COST_CHANGE_FRAC_THRESH)          
            local train_err_below_change_threshold = (self.dTrainErr_frac[nEpochs] < p.TRAIN_ERR_CHANGE_FRAC_THRESH)
            local minTestError_epoch = self:epochOfMinTestError()
            local test_err_stopped_decreasing = (p.TEST_ERR_NEPOCHS_STOP_AFTER_MIN > 0) and (nEpochs - minTestError_epoch) > p.TEST_ERR_NEPOCHS_STOP_AFTER_MIN
            
            --train_err_zero = self.trainErr[nEpochs] == 0
            --test_err_zero = self.testErr[nEpochs] == 0
            
            --print('cost change = ' .. self.dCost_frac[nEpochs] .. '. train change = ' .. self.dTrainErr_frac[nEpochs])
            
            --print('cost below th = ' .. tostring(cost_below_change_threshold) .. '. train below th = ' .. tostring(train_err_below_change_threshold))
            
              
            --print(self.dCost_frac[nEpochs]*100, self.dTrainErr_frac[nEpochs]*100)
            --print(cost_below_change_threshold, train_err_below_change_threshold)
            
            if p.REQUIRE_COST_MINIMUM then
                if train_err_below_change_threshold then
                    print('continuing even though train has bottomed out')
                end
                train_err_below_change_threshold = false
                
                if test_err_stopped_decreasing then
                    print('continuing even though test has stopped improving ')
                end
                test_err_stopped_decreasing = false
            end
            
            
            local decided = false
                        
            --print(nEpochs, p.MIN_EPOCHS, p.MAX_EPOCHS)
            if (nEpochs < p.MIN_EPOCHS) then -- always continue if not yet 10 epochs
                self.satisfiedStopCriterion = false
                self.trainingState = 'SGD'
                decided = true
            end
                --print('too few epochs')
                
            if (nEpochs >= p.MAX_EPOCHS) then -- stop after max
                self.satisfiedStopCriterion = true
                --self.nExtraEpochs = p.EXTRA_EPOCHS+1 -- a hack to stop immediately after this run
                
                self.reason = 'Exceeded max number of epochs (' .. p.MAX_EPOCHS .. ')'               
                decided = true
            end
                        
            if not decided and p.REQUIRE_TRAINING_ERR_MINIMUM and not (train_err_below_change_threshold or test_err_stopped_decreasing) then
                self.satisfiedStopCriterion = false
                self.reason = string.format('Training (or testing) error is still decreasing (changed by %.3f%% in the last epoch)', self.dTrainErr_frac[nEpochs]*100)
                decided = true
            end
            
            if not decided and p.REQUIRE_COST_MINIMUM and not cost_below_change_threshold then
                self.satisfiedStopCriterion = false
                self.reason = string.format('Cost function still decreasing (changed by %.3f%% in the last epoch)', self.dCost_frac[nEpochs]*100)
                print(self.reason)
                decided = true
            end

            if not decided and train_err_below_change_threshold then
                self.satisfiedStopCriterion = true
                self.reason = string.format('Training error changed by only %.3f%% (< threshold of %.3f%%) ', self.dTrainErr_frac[nEpochs]*100, p.TRAIN_ERR_CHANGE_FRAC_THRESH*100 )
            
            elseif not decided and cost_below_change_threshold then
                self.satisfiedStopCriterion = true
                self.reason = string.format('Cost function changed by only %.3f%% (< threshold of %.3f%%) ',
                    self.dCost_frac[nEpochs]*100, p.COST_CHANGE_FRAC_THRESH*100 )
                
            elseif not decided and test_err_stopped_decreasing then
                self.satisfiedStopCriterion = true
                self.reason = string.format('Test error hasnt improved since epoch %d (ie. %d epochs ago, > threshold of %d) ', 
                    minTestError_epoch, nEpochs - minTestError_epoch, p.TEST_ERR_NEPOCHS_STOP_AFTER_MIN )
            end
                
            
            if self.forcingContinue then
                self.satisfiedStopCriterion = false
                self.reason = 'forcing continuation of training...'
            end
            
        end
        --[[
        elseif self.trainingState == 'L-BFGS' then
            -- assume training has completed / converged?
            self.trainingState = 'DONE'
        --]]
        
        
        if self.satisfiedStopCriterion then
            if self.trainingState == 'SGD' and p.SWITCH_TO_LBFGS_AT_END then
                if p.LBFGS_USE_REDUCED_SET_FIRST then
                    self.trainingState = 'L-BFGS-reduced'
                    print('SWITCHING TO L-BFGS (with reduced data set)')
                else
                    self.trainingState = 'L-BFGS'
                    print('SWITCHING TO L-BFGS (using full data set)')
                end
                
            elseif self.trainingState == 'L-BFGS-reduced' then
                self.trainingState = 'L-BFGS'
                print('USING L-BFGS (this time with the complete data set)')
                    
            --elseif self.trainingState == 'L-BFGS' then      
            elseif (self.nExtraEpochs >= p.EXTRA_EPOCHS) or (nEpochs >= p.MAX_EPOCHS) then
                print('COMPLETED TRAINING')
                self.trainingState = 'DONE'
            else
                self.trainingState = 'SGD'
                self.nExtraEpochs = self.nExtraEpochs + 1
            end
        else
            self.trainingState = 'SGD'
        end
        
        local contTraining = self.trainingState ~= 'DONE'
        return contTraining, self.trainingState, self.reason
    end
    
    function TrainingLogger:forceContinue(newTrainingState, contIfBeforeDate)
        newTrainingState = newTrainingState or 'SGD'
        if not ((newTrainingState == 'SGD') or (newTrainingState == 'L-BFGS') or (newTrainingState == 'L-BFGS-reduced')) then
            error('Unknown training state : ' .. newTrainingState)
        end
        
        local tooOld = (contIfBeforeDate == nil) or (self:lastUpdatedTime() < contIfBeforeDate)
        local reason
        if tooOld then
            if not contIfBeforeDate then
                reason = 'contIfBeforeDate = nil'
            else
                reason = 'last updated: ' .. os.date('%x', self:lastUpdatedTime() ) .. ' force after: ' .. os.date('%x', contIfBeforeDate )
            end
        end
        
        if self.satisfiedStopCriterion and tooOld then
            print('==> Force Continue Training...' .. reason)
            self.trainingState = newTrainingState
            self.satisfiedStopCriterion = false
            self.nExtraEpochs = 0
            self.forcingContinue = true
        end
        Self = self
    end
    
    function TrainingLogger:setFile(filename)
        self.filename = filename
        
    end
    
    function TrainingLogger:haveFile()
        return paths.filep(self.filename)
    end
    
    function TrainingLogger:fileDate()
        if paths.filep(self.filename) then
            return lfs.attributes(self.filename).modification
        else
            return nil
        end
    end

    
    function TrainingLogger:loadFromFile()
        tic()
        io.write(string.format('Loading saved trained network from %s ...', paths.basename(self.filename)))
        local loaded_file = nil
                
        local function loadSavedDataFromFile() 
            loaded_file = torch.load(self.filename)        
        end
        local status, result = pcall(loadSavedDataFromFile)
        
        io.write(string.format('[took %d sec]\n', toc()))

        if not status then
            print('Encountered the following error while loading the file : ')
            print(result)
            print('Starting from scratch....')
            assert(string.find(result, 'read error') or string.find(result, 'table index is nil')) 
            return nil
        else
            return loaded_file
        end
            
        --return loaded_file
        
    end
    
    function TrainingLogger:saveToFile()
        if self.params.SAVE_TRAINING then
            torch.save(self.filename, self)
            --print(string.format('Saving training : to file %s\n', self.filename))
        end
        
        local check_save = false
        if check_save then
        
            self_copy = torch.load(self.filename)
            Self = self
            print('Checking saved vs current version ...')
            assert(isequal(self_copy, Self))
            
        end
        
        
    end
        
   

--[[

function torch.load_check(filename, mode)
   mode = mode or 'binary'
   local file = torch.DiskFile(filename, 'r')
   file[mode](file)
   file:clear
   file:quiet()
   local object = file:readObject()
   file:close()
   return object
end

--]]