
if not mycri_kai then
    require 'nn'

    mycri_kai,parent = torch.class('nn.mycri_kai','nn.Criterion')

    -- criterion : get loss
    function mycri_kai:updateOutput(input,target)
        local loss = 0
        --print("my cri getting error...", input:size()[1])
        for i = 1,input:size()[1] do
            if target[i]~=-1 then
                --print('input', input[i], 'target', target[i])
                --io.write(string.format('input_i = %.3f. target_i = %.3f\n', input[i], target[i]))
                loss = loss + 0.5 * math.pow(input[i]-target[i],2)
            else
                --io.write(string.format('input_i = %.3f. target_i = %.3f\n', input[i], target[i]))
                --if input[i]>=-0.51 and input[i]<=0.51 then
                loss = loss + 10 * math.exp(-1*input[i]*input[i]/0.05)
                --[[else
                loss = loss
                end]]--
            --print("targe is -1\n")
        end

    end
    --print("error this time: ",loss)
    self.output = loss
    return self.output
end

-- criterion : get gradients
function mycri_kai:updateGradInput(input,target)
    self.gradInput:resizeAs(input)
    self.gradInput:zero()
    -- for regression, target is a 1D tensor
    for i = 1,input:size()[1] do
        if target[i] ~=-1 then
            self.gradInput[i]=input[i]-target[i]
        else
            --if input[i]>=-0.51 and input[i]<=0.51 then
            -- derivative of a*exp(-(x^2/b))
            self.gradInput[i]=10*(-0.05)*input[i]*math.exp(-1*input[i]*input[i]/0.05)
            --else
            --self.gradInput[i]=0.0
            --end
            --print("targe is -1\n")
        end
    end
    return self.gradInput 
end

end