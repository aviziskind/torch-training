
if not mycri then

    require 'nn'
    mycri,parent = torch.class('nn.mycri','nn.Criterion')

    -- criterion : get loss
    function mycri:updateOutput(input,target)
        local loss = 0
        --print("my cri getting error...", input:size()[1])
        for i = 1,input:size()[1] do
            if target[i]~=-1 then
                loss = loss + 0.5 * math.pow(input[i]-target[i],2)
            else
                --print("targe is -1\n")
            end

        end
        --print("error this time: ",loss)
        self.output = loss
        return self.output
    end

    -- criterion : get gradients
    function mycri:updateGradInput(input,target)
        self.gradInput:resizeAs(input)
        self.gradInput:zero()
        -- for regression, target is a 1D tensor
        for i = 1,input:size()[1] do
            if target[i] ~=-1 then
                self.gradInput[i]=input[i]-target[i]
            else
                --print("targe is -1\n")
            end
        end
        return self.gradInput 
    end

end