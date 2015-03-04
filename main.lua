    
--require('mobdebug').start()

--local home_dir = os.getenv("HOME")
--local home_dir = paths.home

dofile (paths.home .. '/Code/nyu/letters/torch/avi_scripts.lua')

torch.manualSeed(123)

hostname = os.getenv('hostname') 
onLaptop = (hostname == 'XPS')
assert(hostname == 'XPS' or string.find(hostname, '.nyu.edu'))


do 
    --return
end



if onLaptop then
    useGPU = false
else
    useGPU = false
end
useZBSdebugger = false



--if not expName then

--expName = 'ChannelTuning'
--expName = 'Crowding'
--expName = 'Grouping'
expName = 'Complexity'
--expName = 'TrainingWithNoise'
--expName = 'TestConvNet'

modelName = 'ConvNet'
--modelName = 'Texture'
--modelName = 'OverFeat'


if host == 'macbeth' then
    --expName = 'Complexity'
    --modelName = 'Texture'

end



--
--expName = 'TestTrainer'
--expName = 'ReducingCost'
--expName = 'ReducedNetwork'
--expName = 'Test'
--expName = 'TestConvNet'
--expName = 'MetamerLetters'
--expName = 'NoisyLettersTextureStats'



if modelName == 'ConvNet' then
    stimType = 'NoisyLetters'
elseif modelName == 'Texture' then
    stimType = 'NoisyLettersTextureStats'
elseif modelName == 'OverFeat' then
    stimType = 'NoisyLettersOverFeat'
end
    

subfolder_ext = ''
if not onLaptop then
    subfolder_ext = '_NYU'
end


testTest = true
torch.setnumthreads(1)

loadOpts = {}
loadOpts.totalUseFrac = .1
loadOpts.trainFrac  = 0.8   -- fraction of data to use for training (vs testing)
loadOpts.normalizeInputs = false

trainOpts = {}
trainOpts.REQUIRE_COST_MINIMUM = true
trainOpts.REQUIRE_TRAINING_ERR_MINIMUM = true
trainOpts.SWITCH_TO_LBFGS_AT_END = true
trainOpts.LBFGS_USE_REDUCED_SET_FIRST = false
trainOpts.LBFGS_REDUCED_FRAC = 0.1
trainOpts.SAVE_TRAINING = true
trainOpts.BATCH_SIZE = 1

GPU_BATCH_SIZE = 128

if useGPU then
    trainOpts.BATCH_SIZE = GPU_BATCH_SIZE
end

--noisyLetterOpts_str = string.format('_%dori%dx%dy', noisyLetterOpts.Nori, noisyLetterOpts.Nx, noisyLetterOpts.Ny)

------------------------------
if useZBSdebugger then

end
    

letters_dir = paths.home ..'/Code/nyu/letters/' 
torchLetters_dir  = letters_dir .. 'torch/'
matlabLetters_dir = letters_dir .. 'MATLAB/'

dofile (torchLetters_dir .. 'letters_scripts.lua')


results_subfolder = expName .. subfolder_ext.. '/' .. stimType  .. '/' -- .. .. getDataSubfolder({expName = expName, stimType = stimType}) .. '/'
--results_subfolder2 = getDataSubfolder({expName = expName, stimType = stimType}) .. '/'

dataset_subfolder = ''
if expName == 'Crowding' then
    dataset_subfolder = expName .. '/'
end

if onLaptop then
    torch_datasets_dir = torchLetters_dir .. 'datasets/'  .. dataset_subfolder
    matlab_datasets_dir = matlabLetters_dir .. 'datasets/' .. dataset_subfolder
    
    training_dir = torchLetters_dir .. 'TrainedNetworks/' .. results_subfolder
    results_dir = torchLetters_dir .. 'Results/' .. results_subfolder
    
    overfeat_weights_dir = '/usr/local/overfeat/data/default/'

else
    nyu_data_dir = '/misc/vlgscratch2/LecunGroup/ziskind/lettersData/'
    --nyu_data_dir = '/home/ziskind/lettersData/' -- this is a symlink to scratch location
    torch_datasets_dir = nyu_data_dir .. 'torch/datasets/'  .. dataset_subfolder
    matlab_datasets_dir = nyu_data_dir .. 'MATLAB/datasets/'  .. dataset_subfolder
    
    training_dir = nyu_data_dir .. 'torch/TrainedNetworks/' .. results_subfolder
    results_dir = nyu_data_dir .. 'torch/Results/' .. results_subfolder
    
    overfeat_weights_dir = nyu_data_dir .. 'torch/overfeat/data/default/'
    
end

if not paths.dirp(results_dir)     then createFolder(results_dir)     end
if not paths.dirp(training_dir)    then createFolder(training_dir)    end




svhn_dir = torch_datasets_dir .. 'SVHN/'


torch.setdefaulttensortype('torch.FloatTensor')

if useGPU then
    require 'cutorch'
    require 'cunn'    
else
    require 'nn'
end
require 'sys'    
require 'mattorch'
require 'optim'
require 'image' 
require 'lfs'



dofile (torchLetters_dir .. 'logger2.lua')
dofile (torchLetters_dir .. 'trainingLogger.lua')


local actionIfNoGPULock = 'turnGPUoff'
--local actionIfNoGPULock = 'error'

if useGPU then
   
    local GPU_ID = os.getenv('GPU_ID')
    
    if GPU_ID then
        local device_id = GPU_ID + 1
        print(string.format('Detected GPU_ID=%d ==> Selecting GPU #%d', GPU_ID, device_id))
        cutorch.setDevice(device_id)
        assert( cutorch.getDevice(device_id) == device_id)
        cutorch.deviceReset()
        print('done!')
        
    elseif  actionIfNoGPULock == 'error' then
        error('We dont have a lock on any of the GPUs')
    elseif actionIfNoGPULock == 'turnGPUoff' then
        
        print(string.format('NO Detected GPU_ID. Not using GPU'))
        useGPU = false
        trainOpts.BATCH_SIZE = 1
    end
        
end



-- variables needed for loading noisy letters files
--allFontNames = {'Braille', 'Sloan', 'Helvetica', 'Courier', 'Bookman', 'GeorgiaU', 'Yung', 'Kuenstler'}
--allFontNames = {'Braille', 'Sloan', 'Helvetica', 'Bookman', 'Yung', 'Courier', 'Kuenstler'} -- in order of complexity
allFontNames = {'Braille', 'Bookman', 'KuenstlerU', 'Sloan', 'Helvetica', 'Yung', 'Courier'} -- order to see results quickly
allFontNames_ext = {'Braille', 'Checkers4x4',     'Sloan', 'SloanB', 
                    'Helvetica', 'HelveticaB', 'HelveticaU', 'HelveticaUB', 
                    'Courier', 'CourierB', 'CourierU', 'CourierUB', 
                    'Bookman', 'BookmanB', 'BookmanU', 'BookmanUB', 
                    'Yung', 'YungB',     'KuenstlerU', 'KuenstlerUB'}

allSNRs = {0, 1, 1.5,  2,  2.5, 3, 4}

if useOldStim then
    allFontNames = {'Braille', 'GeorgiaU', 'Yung', 'Kuenstler'}
    allSNRs = {0, 1, 2, 3, 4}
end


--[[
if expName == 'Crowding' then
    allFontNames = {'Sloan'}
--    allTDRs = {-2, -1, -0.5, 0} -- {0, .25, 0.5, 0.75, 1}
    allTDRs = {0} -- {0, .25, 0.5, 0.75, 1}
    nTDRs = #allTDRs
    
    loadOpts.trainFrac  = 0.9
    
end
--]]
nFonts = #allFontNames
nSNRs = #allSNRs

--[[
if expName == 'NoisyLettersTextureStats' then
    
    --allFontNames = {'Braille', 'Bookman', 'KuenstlerU', 'Sloan', 'Helvetica', 'Yung', 'Courier'} -- order to see results quickly
    allFontNames = {'Bookman', 'Sloan', 'Helvetica', 'Yung', 'Courier', 'KuenstlerU'} -- order to see results quickly
--'Braille', 
    
end
--]]

local filesToCheckForUpdatesAfterDone = {'main.lua', 'doNoisyLetters.lua', 'doCrowdedLetters.lua'}


print('Loading files')
dofile (torchLetters_dir .. 'load_data.lua')
dofile (torchLetters_dir .. 'generate_model.lua')
dofile (torchLetters_dir .. 'train_model.lua')

dofile (torchLetters_dir .. 'trainingLogger.lua')
--dofile (torchLetters_dir .. 'font.lua')

dofile (torchLetters_dir .. 'lbfgs_cuda.lua')
dofile (torchLetters_dir .. 'lswolfe_cuda.lua')
dofile (torchLetters_dir .. 'doNoisyTrainingBatch.lua')
dofile (torchLetters_dir .. 'doCrowdedTrainingBatch.lua')
dofile (torchLetters_dir .. 'SpatialL1Pooling.lua')

print('done')


local showNoisyExamples = false
if showNoisyExamples then
    local fontName_show = 'Bookman'
    local noisyLetterOpts_show = {Nori = 1, dOri = 0,  Nx = 4, dX = 25, Ny = 1, dY = 0,
                                  sizeStyle = 'k18', stimType = 'NoisyLetters', autoImageSize = false, imageSize = {36,116}, blurStd = 0, noiseFilter = {filterType = 'white', f_exp = 1, cycPerLet_centFreq = 1.3},
                                  --targetPosition = 2, nLetters = 3
                                  }
                           
    ExampleData = loadLetters(fontName_show, 3, noisyLetterOpts_show, loadOpts)
    --exampleData = loadNoisyLetters(allFontNames_ext, {3,4}, loadOpts)
    displayLetterExamples(ExampleData, 100, 1)
    print(ExampleData.descrip)
    return
end
 
 
local showCrowdedExamples = false
if showCrowdedExamples then
    local fontName_show = 'HelveticaUB'
    local sizeStyle_show = 'k18'
    local showSingleLetters = true
    local showDoubleLetters = false
    --local crowdedLetterOpts_show = {xrange = {15,5,75}, targetPosition = {1,2,3,4,5,6,7,8,9,10,11}, logSNR = 4,
      --                              sizeStyle = sizeStyle_show, stimType = 'CrowdedLetters', imageSize = {32, 64}}
    local crowdedLetterOpts_show = {xrange = {15,3,87}, trainTargetPosition = {1,9,13}, logSNR = 4,
                                    sizeStyle = sizeStyle_show, stimType = 'CrowdedLetters', imageSize = {32, 64}}
    local tdr_show = 0
    local distractorSpacing_show = {}
    local all_nDistractors_show = {}
    exampleTrainData1, exampleTestData1 = loadLetters(fontName_show, 4, crowdedLetterOpts_show, loadOpts)
    --exampleData = loadNoisyLetters(allFontNames_ext, {3,4}, loadOpts)
    if showSingleLetters then
        displayLetterExamples(exampleTrainData1, 25, 1)
        print(exampleTrainData1.descrip)
        print(' ');
        --displayLetterExamples(exampleTestData1, 25, 1)
        --print(exampleTestData1.descrip)
        --print(' ');    
    end
    if showDoubleLetters then
        displayLetterExamples(exampleTestData1, 25, 1)
        print(exampleTestData2.descrip)
    end
    return
end



local showNoisyLettersTextureStats = false
if showNoisyLettersTextureStats then
    local fontName_show = 'Bookman'
    local NoisyLettersTextureStatsOpts_show = {Nori = 1, ori_range = 0,  Nx = 1, x_range = 0, Ny = 1, y_range = 0,  --  sizeStyle = ??
                                        stimType = 'NoisyLettersTextureStats',  sizeStyle = 'med',
                                        Nscl_txt = 3, Nori_txt = 2, Na_txt = 3, imageSize = {48,48} }
    loadOpts.normalizeInputs = true
    exampleData = loadNoisyLetters(fontName_show, 3, showNoisyLettersTextureStats, loadOpts)
    --exampleData = loadNoisyLetters(allFontNames_ext, {3,4}, loadOpts)
    --displayLetterExamples(exampleData, 100, 1)
    print(exampleData.descrip)
    return
end
 
if not Models_c then
    Models_c = {}
end

local showMetamerLetterExamples = false
if showMetamerLetterExamples then
    local fontName_show = 'BookmanU'
    local metamerLetterOpts_show = {size = 64, niter = 200, stimType = 'Metamer'}
                           
    local exampleData = loadMetamerLetters(fontName_show, metamerLetterOpts_show, loadOpts)
    --exampleData = loadNoisyLetters(allFontNames_ext, {3,4}, loadOpts)
    displayLetterExamples(exampleData, 100, 1)
    print(exampleData.descrip)
    return
end



if (expName == 'ChannelTuning') or (expName == 'Complexity') or (expName == 'TrainingWithNoise') or (expName == 'Grouping') or (expName == 'TestConvNet') then
    dofile (torchLetters_dir .. 'doNoisyLetters.lua')
    doNoisyLetters() 
end
   

if expName == 'Crowding' then    
    dofile (torchLetters_dir .. 'doCrowdedLetters.lua')
    doCrowdedLetters() 
end


--[[
if expName == 'NoisyLettersTextureStats' then
    
    allFontNames = {'Braille', 'Sloan', 'Helvetica', 'Courier', 'Bookman', 'Yung', 'KuenstlerU'}
    allSNRs = {0, 1, 2, 2.5, 3, 4, 5}
    
    dofile (torchLetters_dir .. 'doNoisyLettersTextureStats.lua')
    doNoisyLettersTextureStats() 
     
   
end
--]]


if expName == 'MetamerLetters' then    
    dofile (torchLetters_dir .. 'doMetamerLetters.lua')
    doMetamerLetters() 
end



--[[
if expName == 'TrainingWithNoise' then    
    dofile (torchLetters_dir .. 'doTrainingWithNoise.lua')
    doTrainingWithNoise() 
                        
end 
--]]

if expName == 'TestConvNet' then
   -- pick a training set that takes a while to converge 
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    trainOpts.REQUIRE_COST_MINIMUM = false
    print('==========now testing convnet==========')
    
    fontName = 'GeorgiaU'
    
    --nStates = {6,16,120} --- ORIGINAL SET ---
--    nStates = {6,16,32} --- ORIGINAL SET ---
    nStates = {6,16} --- ORIGINAL SET ---
    
    snr_train = 4
    --snrs = 
    noisyLetterOpts = {Nx = 1, Ny = 1}
    trainData, testData = loadNoisyLetters(fontName, snr_train, noisyLetterOpts, loadOpts)
    
    networkOpts = {nInputs = trainData.nInputs, nClasses = trainData.nClasses, ConvNet = true, nStates = nStates}
    model_struct = generateModel(networkOpts)
    
    print(string.format('Training network on Font = %s, SNR = %d, # Hidden Units = %s', fontName, snr_train, HiddenUnits_str))      
 
    expSubtitle = getExpSubtitle(fontName, snr_train, networkOpts)
    print('Subtitle = ' .. expSubtitle)

    model_struct = trainModel(model_struct, trainData, testData, trainOpts)
    errRate = testModel(model_struct, testData, verbose)

    print('Err = ' .. errRate)
    

   
end



if expName == 'ReducedNetwork' then
   -- pick a training set that takes a while to converge 
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    
    retrainSecondNetwork = false
    
    fontNames = {'Braille', 'Sloan', 'GeorgiaU', 'Yung', 'Kuenstler'}
--    allNHiddenUnits1 = {10,20,40,80}
  --  allNHiddenUnits2 = {5,6,7,8,9,10}
    allNHiddenUnits1 = {50,100,200}
    if retrainSecondNetwork then
        allNHiddenUnits2 = {5,10,20}
    else
        allNHiddenUnits2 = {5,10,20,40}
    end
    snr_train = 4
    all_snrs_test = {0,1,2,3,4}
    --snrs = 
    results = {}
    results.allNHiddenUnits1 = torch.Tensor(allNHiddenUnits1)
    results.allNHiddenUnits2 = torch.Tensor(allNHiddenUnits2)
    results.all_snrs_test = torch.Tensor(all_snrs_test)
    --results.fontNames = fontNames
    
    for fi, fontName in ipairs(fontNames) do
        trainOpts.SWITCH_TO_LBFGS_AT_END = true
        print(string.format('========== USING Font = %s, SNR = %d, ', fontName, snr_train))
        
        trainData, testData = loadNoisyLetters(fontName, snr_train, noisyLetterOpts, loadOpts)

        --HiddenUnits_str = hiddenLayer_str(nHiddenUnits)
        
        pctCorrect_vH1_H2_snr = torch.Tensor(#allNHiddenUnits1, #allNHiddenUnits2, #all_snrs_test):zero()
    --    --nUnits_tnsr = torch.tensor(allnhiddenunits)
    --    errrates_v_hiddenunits = torch.tensor(nnunits)
    --       
    --    pct_correct_vs_snr = torch.tensor(nnunits, nsnrs):zero()

    --    pct_correct_v_hiddenUnits = torch.Tensor(nnUnits)
        
        
        expSubtitle = getExpSubtitle(fontName, snr_train, 0)

        
        for i1, nHid1 in ipairs(allNHiddenUnits1) do
            networkOpts = {nInputs = trainData.nInputs, nHiddenUnits = nHid1, nClasses = trainData.nClasses, doSplit = true}
            model_struct1 = generateModel(networkOpts)
            
            print(string.format('[[ %d ]]Training primary network on # Hidden Units = %d', i1, nHid1))      

            --trainOpts.MAX_EPOCHS = 10
            trainModel(model_struct1, trainData, testData, trainOpts)
            
            for i2, nHid2 in ipairs(allNHiddenUnits2) do
                
                print(string.format(' (%d) Now copying to network with # Hidden Units = %d', i2, nHid2))      
         
                networkOpts2 = {nInputs = trainData.nInputs, nHiddenUnits = nHid2, nClasses = trainData.nClasses, doSplit = true}
                model_struct2 = generateModel(networkOpts2)

                errRate1_a = testModel(model_struct1, testData, verbose)
                errRate2_a = testModel(model_struct2, testData, verbose)
                print('Initial errors are : ' .. errRate1_a .. ' and ' .. errRate2_a)
                
                copyModelUnits(model_struct1, model_struct2)
                errRate1_a = testModel(model_struct1, testData, verbose)
                errRate2_a = testModel(model_struct2, testData, verbose)
                print('Errors after copying : ' .. errRate1_a .. ' and ' .. errRate2_a)

                if retrainSecondNetwork then
                    trainOpts.SWITCH_TO_LBFGS_AT_END = true
                    trainOpts.freezeFeatures = true

                    print('Retraining model 2 after freezing (reduced) feature layer')
                    trainModel(model_struct2, trainData, testData, trainOpts)
                end
                errRate2_final = testModel(model_struct2, testData, verbose)
                pctCorrect_vH1_H2_snr[i1][i2] = errRate2_final
            
                print('So far ... ')
                print(pctCorrect_vH1_H2)
                
                for si,snr_test in ipairs(all_snrs_test) do
                    _, testData_noisy = loadNoisyLetters(fontName, snr_test, noisyLetterOpts, loadOpts)
                    
                    pctCorrect_i = 100 - testModel(model_struct2, testData_noisy)
                    pctCorrect_vH1_H2_snr[i1][i2][si] = pctCorrect_i
                    
                    print(string.format(' -- testing on snr = %.1f: (%.1f %% correct)', snr_test, pctCorrect_i));
                    
                end
                
                
            end
        end
        print(pctCorrect_vH1_H2_snr)
        results[fontName] = pctCorrect_vH1_H2_snr
     
    end 
    
    fn_save = 'ReducedModels'
    if not retrainSecondNetwork then
        fn_save = fn_save .. '_noRetrain'
    end
    mattorch.save(results_dir .. fn_save .. '.mat', results)
 
 
            
            --visualizeHiddenUnits(model)
        
end






if expName == 'ReducingCost' then    
    

    fontName = 'Kuenstler'
    --fontName = 'Braille'
    
    snr_train = 4
    test_snrs = { 0, 1, 2, 3, 4 }
---[[
    nHiddenUnits = 5
       
    nsnrs = #test_snrs
    --nUnits_tnsr = torch.Tensor(allHiddenUnits)
    errRates_v_snr = torch.Tensor(nsnrs)
   
   
    trainData = {}
    testData = {}
    testData_noise = {}
    models = {}
    criteria = {}
    trainOpts = {}
    errRates_v_snr_train_cost = {}
    errRates_v_snr_train_cost_LBFGS = {}
    errRates_v_snr_cost_LBFGS = {}
    
    
    trainData,testData = loadNoisyLetters(fontName, snr_train, loadOpts)
    
    errRates_snr_v_snr = torch.Tensor(nsnrs, nsnrs):zero()
   
    torch.manualSeed(123)
    print('Training until cost or train err flattens out, no L-BFGS')               
    trainOpts.REQUIRE_COST_MINIMUM = false 
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    expSubtitle = (getExpSubtitle(fontName, snr_train, nHiddenUnits) .. '_train_cost')
    
    networkOpts = {nInputs = trainData.nInputs, nHiddenUnits = nHiddenUnits, nClasses = trainData.nClasses}
        
    models[1] = generateModel(networkOpts)
    trainModel(models[1], trainData, testData, trainOpts)
    
    torch.manualSeed(123)
    print('Training until cost or train err flattens out, Switching to L-BFGS at the end')
    expSubtitle = (getExpSubtitle(fontName, snr_train, nHiddenUnits) .. '_train_cost_LBFGS')    
    
    models[2] = generateModel(networkOpts)
    trainModel(models[2], trainData, testData, trainOpts)
            
            --[[
    torch.manualSeed(123)
    print('Training until on cost flattens out, switching to LBFGS at the end')
    trainOpts.REQUIRE_COST_MINIMUM = true
    trainOpts.SWITCH_TO_LBFGS_AT_END = true
    expSubtitle = (getExpSubtitle(fontName, snr_train, nHiddenUnits) .. '_cost_LBFGS')    
    models[3], criteria[3] = generateModel(trainData.nInputs, nHiddenUnits, trainData.nClasses)
    models[3] = trainModel(models[3], criteria[3], trainData, testData, trainOpts)           
        --]]
        
    for s_i,test_snr in ipairs(test_snrs) do
        print('Loading SNR = ' .. test_snr)        
        _, testData_noise[s_i] = loadNoisyLetters(fontName, test_snr, loadOpts)
    end
    
    for s_i,snr_to_do in ipairs(test_snrs) do
        errRates_v_snr_train_cost[s_i] = testModel(models[1], testData_noise[s_i])
        errRates_v_snr_train_cost_LBFGS[s_i] = testModel(models[2], testData_noise[s_i])
        --errRates_v_snr_cost_LBFGS[s_i]= testModel(models[3], testData_noise[s_i])
    end
        
    print('Results for training until cost or train err flattens out, no L-BFGS')  
    print(errRates_v_snr_train_cost)
    print('Results for training until on cost flattens out, switching to LBFGS at the end')
    print(errRates_v_snr_train_cost_LBFGS)
    --print('Results for training until on cost flattens out, switching to LBFGS at the end')
    --print(errRates_v_snr_cost_LBFGS)
        
        --errRate = testModel(model, testData, verbose)
        --errRates_v_snr[s_i] = errRate
        
end 



if expName == 'TestSplitNetwork' then
   -- pick a training set that takes a while to converge 
    trainOpts.SWITCH_TO_LBFGS_AT_END = false
    
    
    fontName1 = 'GeorgiaU'
    fontName2 = 'Kuenstler'
    nHiddenUnits1 = 20
    nHiddenUnits2 = 5
    snr_train = 4
    --snrs = 
    
    pretrainData, pretestData = loadNoisyLetters(fontName1, snr_train, loadOpts)
    retrainData, retestData = loadNoisyLetters(fontName2, snr_train, loadOpts)

    HiddenUnits_str = hiddenLayer_str(nHiddenUnits)
--    --nUnits_tnsr = torch.tensor(allnhiddenunits)
--    errrates_v_hiddenunits = torch.tensor(nnunits)
--       
--    pct_correct_vs_snr = torch.tensor(nnunits, nsnrs):zero()

--    pct_correct_v_hiddenUnits = torch.Tensor(nnUnits)
    
    networkOpts1 = {nInputs = pretrainData.nInputs, nHiddenUnits = nHiddenUnits1, nClasses = pretrainData.nClasses, doSplit = true}
    networkOpts2 = {nInputs = pretrainData.nInputs, nHiddenUnits = nHiddenUnits2, nClasses = pretrainData.nClasses, doSplit = true}
     
    model_struct1 = generateModel(networkOpts1)
    model_struct2 = generateModel(networkOpts2)
      
    print(string.format('Pretraining network on Font = %s, SNR = %d, # Hidden Units = %s', fontName1, snr_train, HiddenUnits_str))      
 
    expSubtitle = getExpSubtitle(fontName, snr_train, nHiddenUnits)

    --trainOpts.MAX_EPOCHS = 2

    model_struct1 = trainModel(model_struct1, pretrainData, pretestData, trainOpts)
    errRate1_a = testModel(model_struct1, pretestData, verbose)
    errRate2_a = testModel(model_struct2, pretestData, verbose)
    --errRate2_a = testModel(model_struct,  retestData, verbose)
    print('Err1 = ' .. errRate1_a)
    print('Err2 = ' .. errRate2_a)
    
    print('Copying network 1 to 2')
    copyModelUnits(model_struct1, model_struct2)
    
    errRate1_a = testModel(model_struct1, pretestData, verbose)
    errRate2_a = testModel(model_struct2, pretestData, verbose)
    --errRate2_a = testModel(model_struct,  retestData, verbose)
    print('Err1 = ' .. errRate1_a)
    print('Err2 = ' .. errRate2_a)
    
    print('Retraining classifier of network 2 (with features frozen)')

    trainOpts.MAX_EPOCHS = 5
    trainOpts.freezeFeatures = true
    model_struct2 = trainModel(model_struct2, pretrainData, pretestData, trainOpts)
    
    errRate2_a = testModel(model_struct2, pretestData, verbose)
    print('Err2 = ' .. errRate2_a)
    
    --editModel(model_struct, 4)
    if true then
        return
    end
    model_struct = trainModel(model_struct, pretrainData, pretestData, trainOpts)

    
    print(string.format('Now retraining only upper classifier layer on Font = %s, SNR = %d, # Hidden Units = %s', fontName2, snr_train, nHiddenUnits_str))      
    trainOpts.freezeFeatures = false

    model_struct = trainModel(model_struct, retrainData, retestData, trainOpts)

    
    errRate1_a = testModel(model_struct, pretestData, verbose)
    errRate2_a = testModel(model_struct,  retestData, verbose)
    print('Err1 = ' .. errRate1_a)
    print('Err2 = ' .. errRate2_a)
    --pct_correct_v_hiddenUnits[h_i] = 100 - errRate
    
            
            --visualizeHiddenUnits(model)
        
end


if expName == 'Test' then
  

    fontName = 'GeorgiaU'
    
    snr_train = 2
    --allsnrs = { 0, 1, 2, 3, 4 }
    --nsnrs = #allsnrs
    
    nHiddenUnits = 3
       
    --nUnits_tnsr = torch.Tensor(allHiddenUnits)
    --errRates_v_snr = torch.Tensor(nsnrs)
   
    print('Loading ... ')
    trainData, testData = loadNoisyLetters(fontName, snr_train, {}, loadOpts)
    
     
    expSubtitle = getExpSubtitle(fontName, snr_train, nHiddenUnits)
    
    networkOpts = {nInputs = trainData.nInputs, nHiddenUnits = nHiddenUnits, nClasses = trainData.nClasses}
    print('Generate ... ')
    model_struct = generateModel(networkOpts)
    trainOpts.freezeFeatures = false
    trainOpts.MAX_EPOCHS = nil
    
    print('Training ... ')
    model_struct = trainModel(model_struct, trainData, testData, trainOpts)           
    
    errRate = testModel(model_struct, testData, verbose)
    
    print('Error rate = ' .. errRate)
        
    --errRate = testModel(model, testData, verbose)
    --errRates_v_snr[s_i] = errRate
        
    
        
end








--[[
sgd_params = {
    learningRate = 1e-3,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0
}

lbfgs_params = {
   lineSearch = optim.lswolfe,
   maxIter = 100,
   verbose = true
}


--[[
doSGD = true
doLBFGS = false

if doSGD then
    print('')
    print('============================================================')
    print('Training with SGD')
    print('')

    
    timer = torch.Timer()
    batchSize = 1
    trainUsingSGD()

    nSec = timer:time().real; 
    print(nSec .. ' seconds for SGD;' ..  nSec/nEpochsDone ..  ' sec average per epoch' )
    
    --timer:reset()
    print(getPercentCorrect())
end

if doLBFGS then
    print('============================================================')
    print('Training with L-BFGS')

    timer:reset()
    batchSize = m
    _,fs = optim.lbfgs(feval,x,lbfgs_params)
    print(timer:time().real .. ' seconds for L-BFGS')
    print(getPercentCorrect())

end
-- fs contains all the evaluations of f, during optimization

--print('history of L-BFGS evaluations:')
--print(fs)


--]]






--[[
torch_dir = '/home/avi/Code/torch/letters/'
stop_file = torch_dir .. '_stop'
stopping_file = torch_dir .. '_stopping'
stopped_file = torch_dir .. '_sto'
if paths.filep(stopping_file) then
    os.rename(stopping_file, stopped_file)
    error('Stop');
end
--]]
--if paths.filep(stop_file) then
  --  os.rename(stop_file, stop_file_tmp)
--end


--[[
useOldStim = false

    if not useOldStim then
        noisy_letters_dir = torch_datasets_dir .. 'NoisyLetters/'
    else
        noisy_letters_dir = torch_datasets_dir .. 'NoisyLetters_orig/'
    end
--]]


---[[
--cmd = torch.CmdLine()
--cmd:option('-THREAD_ID', -1, 'max iterations')
--cmd:option('-N_CPUs', 8, 'max iterations')
--opt = cmd:parse(arg or {})
--print(opt)
--require 'sys'
--t = os.time()
--print('pausing ...')
--while (os.time() - t < 1) do  
--end

--]]

--[[
--require 'os'

--]]