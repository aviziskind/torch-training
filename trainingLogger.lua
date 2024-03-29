

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
        self.params.TEST_ERR_NEPOCHS_STOP_AFTER_MIN = 2
        self.params.MIN_EPOCHS = 10
        self.params.MAX_EPOCHS = 500
        self.params.EXTRA_EPOCHS = 1 -- after satisfies stopping criteria -- do another few epochs
        self.params.SWITCH_TO_LBFGS_AT_END = false
        self.params.LBFGS_USE_REDUCED_SET_FIRST = false
        self.params.REQUIRE_COST_MINIMUM = false -- require cost function to reach local 
        self.params.REQUIRE_TRAINING_ERR_MINIMUM = true
        self.params.STOP_IF_ZERO_TEST_ERROR = true
        self.params.SAVE_TRAINING = true
                                  -- minimum (vs use either cost/train error as criterion)
        
        
        self.trainingClassifier = true;
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
            S1 = self
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
      -- but, also have a minimum of MIN_EPOCHS (unless STOP_IF_ZERO_TEST_ERROR = true, and have no test error)
      
      --. and if gradient started off small, wait until got at least twice max gradient of beginning, before allowing to settle down.
      -- is if  gradient that is greater than initial gradient 
      -- keep 

        
        local nEpochs = self.nEpochs
                    
        local cost_below_change_threshold, train_err_below_change_threshold, minTestError_epoch, test_err_stopped_decreasing
    
        p.nEpochsAgoSatisfiedCriterion = 0;
                    
        if nEpochs > 1  then -- and self.trainingState == 'SGD' then
            self.dCost_frac = self.dCost_frac or {}
            
            self.dCost_frac[nEpochs] = self.dCost[nEpochs] / (self.cost[nEpochs-1] + 1e-5)
            cost_below_change_threshold = (self.dCost_frac[nEpochs] < p.COST_CHANGE_FRAC_THRESH)          
            --io.write(string.format(' dCost = %.5f  <?  th = %.5f  :  %s \n ', 
              --      self.dCost_frac[nEpochs], p.COST_CHANGE_FRAC_THRESH, tostring(cost_below_change_threshold)))
            S = self
            --print('cost_below_change_threshold', cost_below_change_threshold)
            
                self.dTrainErr_frac = self.dTrainErr_frac or {}
            
                self.dTrainErr_frac[nEpochs] = self.dTrainErr[nEpochs] / (self.trainErr[nEpochs-1] + 1e-5)

            --if p.trainingClassifier then            
                --print(self.dTrainErr_frac[nEpochs] .. ' = ' ..  self.dTrainErr[nEpochs] .. ' / ' .. self.trainErr[nEpochs-1])

                --nAverage = 1
                --iStart = math.max(1, nEpochs-nAverage)

                --for i = iStart, nEpochs do
                --    self.dcost_frac_rav[nEpochs] = self.dcost_frac_rav[nEpochs] +  self.dcost_frac[i]
                --end
                --self.dcost_frac_rav[nEpochs] = self.dcost_frac_rav[nEpochs] / (nEpochs - iStart + 1)
                
                train_err_below_change_threshold = (self.dTrainErr_frac[nEpochs] < p.TRAIN_ERR_CHANGE_FRAC_THRESH)
                minTestError_epoch = self:epochOfMinTestError()
                test_err_stopped_decreasing = (p.TEST_ERR_NEPOCHS_STOP_AFTER_MIN >= 0) 
                                        and (nEpochs - minTestError_epoch > p.TEST_ERR_NEPOCHS_STOP_AFTER_MIN)
                                        
                p.test_err_stopped_decreasing = test_err_stopped_decreasing
                if test_err_stopped_decreasing then
                    p.nEpochsAgoSatisfiedCriterion = nEpochs - minTestError_epoch - p.TEST_ERR_NEPOCHS_STOP_AFTER_MIN
                end
                print(string.format('    [Min Test err epoch = %d. cur epoch = %d]. ', minTestError_epoch, nEpochs))
                
        --end
        --print('TEST_ERR_NEPOCHS_STOP_AFTER_MIN = ', p.TEST_ERR_NEPOCHS_STOP_AFTER_MIN)
        --print('test_err_stopped_decreasing = ', test_err_stopped_decreasing);
        
            --train_err_zero = self.trainErr[nEpochs] == 0
            --test_err_zero = self.testErr[nEpochs] == 0
            
            --print('cost change = ' .. self.dCost_frac[nEpochs] .. '. train change = ' .. self.dTrainErr_frac[nEpochs])
            
            --print('cost below th = ' .. tostring(cost_below_change_threshold) .. '. train below th = ' .. tostring(train_err_below_change_threshold))
            
              
            --print(self.dCost_frac[nEpochs]*100, self.dTrainErr_frac[nEpochs]*100)
            --print(cost_below_change_threshold, train_err_below_change_threshold)
            
            --[[
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
            --]]
            
            local decided = false
                        
            -- 0. manual flag
            if self.forcingContinue then
                self.satisfiedStopCriterion = false
                self.reason = 'forcing continuation of training...'
                decided = true
            end
            
            -- 1. use min/max epochs
            
            if (nEpochs < p.MIN_EPOCHS) then -- always continue if not yet MIN_EPOCHS epochs
                self.satisfiedStopCriterion = false
                self.reason = 'too few epochs'
                self.trainingState = 'SGD'
                decided = true
            end
                --print('too few epochs')
                
            if (nEpochs >= p.MAX_EPOCHS) then -- always stop after MAX_EPOCHS
                self.satisfiedStopCriterion = true
                --self.nExtraEpochs = p.EXTRA_EPOCHS+1 -- a hack to stop immediately after this run
                
                self.reason = 'Exceeded max number of epochs (' .. p.MAX_EPOCHS .. ')'               
                decided = true
            end
            
            
            
            -- 2. conditions for training a classifier (have trainining & test error)
            if p.trainingClassifier then
                
                -- if test error is 0 --> STOP
                if not decided and (self.testErr[nEpochs] == 0 and p.STOP_IF_ZERO_TEST_ERROR ) then 
                    self.satisfiedStopCriterion = true
                    self.reason = 'training a classifier and test error is zero'
                    self.trainingState = 'SGD'
                    decided = true
                
                end
                
                
                -- if test error stopped decreasing over last N epochs --> STOP
                if not decided and test_err_stopped_decreasing then
                    self.satisfiedStopCriterion = true
                    self.reason = string.format('Test error hasnt improved since epoch %d (ie. %d epochs ago, > threshold of %d) ', 
                        minTestError_epoch, nEpochs - minTestError_epoch, p.TEST_ERR_NEPOCHS_STOP_AFTER_MIN )
                    
                    decided = true
                end
                
                
                -- if training error stopped decreasing --> STOP
                if not decided and train_err_below_change_threshold then --and not p.REQUIRE_TRAINING_ERR_MINIMUM then
                    self.satisfiedStopCriterion = true
                    self.reason = string.format('Training error changed by only %.3f%% (< threshold of %.3f%%) ', self.dTrainErr_frac[nEpochs]*100, p.TRAIN_ERR_CHANGE_FRAC_THRESH*100 )
                    decided = true
                end
          
          
                --> training error still decreasing --> CONTINUE
                if not decided and p.REQUIRE_TRAINING_ERR_MINIMUM and not 
                    (train_err_below_change_threshold or (self.trainingClassifier and test_err_stopped_decreasing)) then
                    self.satisfiedStopCriterion = false
                    self.reason = string.format('Training (or testing) error is still decreasing (changed by %.3f%% in the last epoch)', self.dTrainErr_frac[nEpochs]*100)
                    decided = true
                end
            
            end
        
            --2. conditions for training a regression model (no training & test error) OR a classifier
            --if not p.trainingClassifier then -- eg regresssion
            
            --> cost stopped decreasing --> STOP
            if not decided and p.REQUIRE_COST_MINIMUM and cost_below_change_threshold then  -- have stopped decreasing
                self.satisfiedStopCriterion = true
                self.reason = string.format('Cost function has stopped decreasing (changed by only %.3f%% in the last epoch)',
                    self.dCost_frac[nEpochs]*100)
                print(self.reason)
                decided = true
            end

            --> cost still decreasing --> CONTINUE
            if not decided and p.REQUIRE_COST_MINIMUM and not cost_below_change_threshold then -- still decreasing
                self.satisfiedStopCriterion = false
                self.reason = string.format('Cost function still decreasing (changed by %.3f%% in the last epoch)',
                    self.dCost_frac[nEpochs]*100)
                print(self.reason)
                decided = true
            end
                
            
            --[[
            print('decided', decided)
            print('REQUIRE_COST_MINIMUM', p.REQUIRE_COST_MINIMUM)            
            print('cost_below_change_threshold', cost_below_change_threshold)
            error('!')
--]]
            if not decided and cost_below_change_threshold then
                self.satisfiedStopCriterion = true
                self.reason = string.format('Cost function changed by only %.3f%% (< threshold of %.3f%%) ',
                    self.dCost_frac[nEpochs]*100, p.COST_CHANGE_FRAC_THRESH*100 )
                decided = true
            end
            
  
        end
        --[[
        elseif self.trainingState == 'L-BFGS' then
            -- assume training has completed / converged?
            self.trainingState = 'DONE'
        --]]
        
        
        if self.satisfiedStopCriterion then
            self.nExtraEpochs = math.max(self.nExtraEpochs, p.nEpochsAgoSatisfiedCriterion)
            
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
		local maxNTries = 3
		local nTries = 0
		 
        local function loadSavedDataFromFile() 
            loaded_file = torch.load(self.filename)        
        end
		
		local status, result = false, nil
		while (nTries < maxNTries) and (status == false) do
		
			status, result = pcall(loadSavedDataFromFile)

			if not status then -- failed
				io.write(string.format('Status = "%s"\n', result))
				if (result == 'stop') or string.find(result, 'interrupted!') then
					print('Received stop signal from user. Aborting....');
					error('Aborted by user')
				end
				local sec_wait = 5 + ( torch.random() % 10 )
				print(string.format('Load failed, trying again in %s seconds', sec_wait))
				sys.sleep(sec_wait)
				nTries = nTries + 1
			end
		
		end
		
		if not status then
			print('Encountered the following error while loading the file : ')
			print(result)
			print('Starting from scratch....')
			assert(string.find(result, 'read error') or string.find(result, 'table index is nil')) 
			return nil
		else
			io.write(string.format('[took %d sec]\n', toc()))
			return loaded_file
		end
		
        
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