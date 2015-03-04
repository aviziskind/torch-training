---[[



--------------------------------------
expName = 'Complexity'
for ti = 1,10 do
    
    nTrials = ti
    setName = 'nStates'
    dofile 'main.lua'

    setName = 'filtSizes'
    dofile 'main.lua'
                                             
    setName = 'poolSizes' 
    dofile 'main.lua'
        
    setName = 'poolTypes'
    dofile 'main.lua'

end

--expName = 'Crowding'
--dofile 'main.lua'


--]]

--expName = 'Complexity'

--[[
print('======================================================')
print('======================= SMALL ========================')
sizeStyle = 'sml'
dofile 'main.lua'

print('======================================================')
print('======================= MEDIUM ========================')
sizeStyle = 'med'
dofile 'main.lua'

print('======================================================')
print('======================= DEFAULT ========================')
sizeStyle = 'dflt'
dofile 'main.lua'

print('======================================================')
print('======================= BIG ========================')
sizeStyle = 'big'
dofile 'main.lua'

--]]