from re import S
import numpy as np
import torch
# import torch.nn as nn
# import torch.functional as F
from CRNN import FNN
from scipy.stats import norm
# from torch.nn.functional import normalize
import matplotlib.pyplot as plt
import math
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil
import os

alpha = 10.0
beta = 0.001
network = FNN(n_classes_in=7, n_classes_out=3)
network.load_state_dict(torch.load('./models/model_3_epoch8.pth'))

def sigmoid(x, c = 1):
  return 1 / (1 + math.exp(-c*x))


def get_age_band(age):
    if age <= 35:
        return np.exp(3)
    elif age <= 55:
        return np.exp(2)
    return np.exp(1)

def get_score(score):
    if score == 'Top 0-20%':
        return 100
    elif score == 'Top 20-40%':
        return 70
    elif score == 'Top 40-60%':
        return 50
    elif score == 'Top 60-80%':
        return 30
    return 10

def get_edu(edu):
    if edu == 'High school':
        return 1
    elif edu == 'College':
        return 2
    elif edu == 'Associate degree':
        return 3
    elif edu == 'Bachelors degree':
        return 4
    elif edu == 'Masters degree':
        return 5
    else:
        return 6

def get_scale(x):
    if x == 'Always':
        return 9
    elif x == 'Very often':
        return 7
    elif x == 'Sometimes':
        return 5
    elif x == 'Rarely':
        return 3
    return 1

def get_assess_type(assess_type):
    if assess_type == 'CMA':
        return 1
    return -1

def get_analysis(miu):
    analysis = {}
    advice = {}
    key1 = 'Achievement'
    if miu[0] < 1/3:
        value1 = 'The “Achievement” index contributes largely to your motivation. Your will to achieve personal realization in work, studies, or other project motivates you to continue and enjoy in the virtual learning environment.'
        advice1 = 'Achievement is usually an important part of personal motivation. You should seek for the advantages and good qualities of yourself more and pump yourself up. Or, if the other way around, be more confident of your achievements. Confidence and self-awareness is one of the most valuable aspect!\n'
    elif miu[0] < 2/3:
        value1 = 'Your “achievement” index does not contribute greatly to your motivation, but it offers a medium fraction to your motivating power. You enjoy the process of gaining a sense of achievement and contentment.'
        advice1 = "Being able to realize your achievements' values and qualities is great. Keep yourself motivated, and don't get over-confident in uncontrollable factors such as tests or grades! Be confident and realistic."
    else:
        value1 = 'Contributing to overall motivation, your “achievement” ranks the lowest. You either might not feel to be most strongly motivated by your excellent achievements, or do not value academic or personal achievements that much.'
        advice1 = "Being able to realize your achievements' values and qualities is great. Keep yourself motivated, and don't get over-confident in uncontrollable factors such as tests or grades! Be confident and realistic."
    analysis[key1] = value1
    advice[key1] = advice1
    key2 = 'Affiliation'
    if miu[1] < 1/3:
        value2 = 'Your “affiliation" index is relatively high in contributing to your motivation in stay-at-home learning. This is rather rare, since people usually feel isolated staying at home. However, this is a good sign in maintaining your social relationships as well as community-minded learning.'
        advice2 = 'Low affiliation is not something to worry about. In fact, many who have low affiliation also persisted through their VLE experience through self-motivation. However, if you want to progress together with your peers, you may concentrate on socializing your academic interest and conduct group projects or researches.'
    elif miu[0] < 2/3:
        value2 = 'Your “affiliation” index is medium, which means you are doing a good job in maintaining social relationships, but to blend in with your online community is no longer your priority motivation for learning and progressing.'
        advice2 = 'Being able to absorb energy and turn them into motivating-power is great. You enjoy equally conducting group of individual projects. Just continue your footsteps!'
    else:
        value2 = 'You have low willingness to join and conform to the community around you. Staying at home is hard to find motivation primarily from your social environment, and you seek motivation from other sources mainly.'
        advice2 = 'This is a great sign of your ability to both thrive in a community and absorb motivating-power from the community, which often accompanies leadership. You might enjoy group projects just as individual projects, as you feel prompted to do better in a group-working environment, as well as enjoying the process. Continue on what motivates you, but do note that overly relying on peers or mentors’ encouragement for motivation for a long time can be fragile.'
    analysis[key2] = value2
    advice[key2] = advice2
    key3 = 'Power'
    if miu[1] < 1/3:
        value3 = 'Your will to contribute and make positive influence to the community or those around you is very strong. It is the most important belief that continues to support and drive you in your academic setting.'
        advice3 = 'You should often think more into the meaning of your work. For instance, why does it matter? What impact can it bring to the community or society? No matter art or science, an important part of all academic studies is their beauty in bringing benefits to the world. If you realize the different aspects of the purpose of your hard work, you will feel much more motivated! VLE is definitely not a limiting factor to the “power” you can have to the world in the future.'
    elif miu[0] < 2/3:
        value3 = 'You realize the importance of making an impact and want to commit yourself in using what you learned to solve real-world problems in the future. However, at the current stage, you do not have enough motivation that contributes to the “power” index, and are primarily motivated by other factors.'
        advice3 = 'You are in good place for the “power” category: you are using future expectations of what you can do using the subjects you are learning or the projects you are doing to push yourself. Too low or high power can be risky to one’s motivation in academic pursuit, but you are doing just right. VLE is definitely not a limiting factor to the “power” you can have to the world in the future.'
    else:
        value3 = 'You are more focused on your learning experiences and acquiring of knowledge as opposed to a strong wish to apply the knowledge to make an impact at the current stage. This can be a result of a lack of awareness of making a powerful impact due to long-term virtual learning.'
        advice3 = 'It is fantastic that you are aware of the different use cases or applications of the subjects you are studying or the research projects you are doing. Keep in mind that overly focused on your influence on the world and the impact you can give might lead to an overestimate your goals and expectations in the long-run.'
    analysis[key3] = value3
    advice[key3] = advice3
    return analysis, advice



def gen_survey_result(params):
    result = {}
    try:
        age = int(params.get("age")) 
    except Exception:
        age = 17
    try:
        interact_times = float(params.get("interact_times")) 
    except Exception:
        interact_times = 3.2
    try:
        anxiety = int(params.get("stress_index"))
        contentment = int(params.get("hobbies_index"))
        social_contentment = int(params.get("social_index"))
        academic_contentment = int(params.get("academic_index"))
        comp_happiness = int(params.get("comparison_index"))
        compliments = int(params.get("compliments_index"))
        happiness = int(params.get("happy_index"))
        procrastination = int(params.get("procrastinate_index"))
        finance = int(params.get("finance_index"))
        depression = params.get("depressed_status")
        loneliness = params.get("lonely_status")
        edu = params.get("education_level")
        score = params.get("academic_score")
        assessment_type = params.get("assessment_type")
    except Exception as e:
        print("Error happened in processing the parameters: {}".format(e))
        return {"message": "Failed: Error parsing inputs"}

    z = (finance*10-49.52)/28.17
    band_score = norm.cdf(z)

    ex_env = alpha*(1 if assessment_type == 'CMA' else -1) + np.exp(finance/10)
    ex_phy = get_age_band(age) + 12.178
    inp = torch.Tensor([get_assess_type(assessment_type),
                    finance*10,
                    get_age_band(age),
                    get_edu(edu),
                    (get_score(score)-10)/80*500+1500,
                    sigmoid(-np.log(get_score(score))+academic_contentment+social_contentment+contentment-anxiety+compliments-procrastination-get_scale(depression)+30, 0.5)*6, 
                    5*sigmoid(academic_contentment-np.log(procrastination)+get_score(score)+compliments+social_contentment+band_score-abs(comp_happiness-5)+5-get_scale(loneliness)-anxiety, 0.01)*60])
    out = network(inp)
    in_ach = out[0]
    in_aff = out[1]
    in_pow = out[2]

    in_inp = torch.Tensor([[in_ach, in_aff, in_pow]])
    in_inp = in_inp.view(in_inp.size(0), -1)
    in_inp -= in_inp.min(1, keepdim=True)[0] 
    in_inp /= in_inp.max(1, keepdim=True)[0] 
    in_inp = in_inp[0]

    ex_inp = torch.Tensor([sigmoid(ex_env), sigmoid(ex_phy)])

    r = torch.mean(in_inp)*torch.mean(ex_inp)
    punishment = torch.Tensor([abs(np.sin(6*np.pi*in_inp[1].item()*in_inp[2].item())), abs(np.sin(6*np.pi*in_inp[0].item()*in_inp[2].item())), abs(np.sin(6*np.pi*in_inp[0].item()*in_inp[1].item()))])
    reward = torch.Tensor([np.tanh(in_inp[0].item())+(ex_inp[0].item()+ex_inp[1].item())*in_inp[0].item()/2, np.tanh(in_inp[1].item())+(ex_inp[0].item()+ex_inp[1].item())*in_inp[1].item()/2, np.tanh(in_inp[2].item())+(ex_inp[0].item()+ex_inp[1].item())*in_inp[2].item()/2])
    net_reward = (reward - punishment)
    miu = torch.Tensor([r,r,r])/net_reward
    miu = torch.Tensor([sigmoid(miu[0].item()), sigmoid(miu[1].item()), sigmoid(miu[2].item())])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, punishment[0].item(), punishment[1].item(), punishment[2].item(), color='b')
    ax.quiver(0, 0, 0, reward[0].item(), reward[1].item(), reward[2].item(), color='r')
    ax.quiver(0, 0, 0, miu[0], miu[1], miu[2], color='g')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.title("Reward, Punishment, and Physical Motivation Vectors")
    ax.set_xlabel("Achievement")
    ax.set_ylabel("Affiliation")
    ax.set_zlabel("Power")
    X, Y, Z = axes3d.get_test_data(0.05)
    def animate(i):
        ax.view_init(elev=30., azim=3.6*i)
        return fig,
    ani = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=True)    
    ani.save('vectors.html', writer = 'html', fps = 30)
    result.update({"vector": "vectors.html"})
    if os.path.exists("static/vectors.html"):
        os.remove("static/vectors.html")
    if os.path.exists("static/vectors_frames"):
        shutil.rmtree("static/vectors_frames")
    shutil.move("vectors.html", "static/vectors.html")
    shutil.move("vectors_frames", "static/")

    analysis, advice = get_analysis(miu)
    result.update({"analysis": analysis})
    result.update({"advice": advice})
    # print("Result:{}".format(result))

    categories = ['Physical Conditions', 'Environment', 'Achievement', 'Affiliation', 'Power']
    categories = [*categories, categories[0]]
    # items = [ex_inp[0]*0.8+0.2, ex_inp[1]*0.8+0.2, miu[0]*0.8+0.2, miu[1]*0.8+0.2, miu[2]*0.8+0.2]
    items = [ex_inp[0], ex_inp[1], miu[0], miu[1], miu[2]]
    items = [*items, items[0]]
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(items))

    fig = plt.figure(figsize=(9, 8))
    plt.subplot(polar=True)
    plt.plot(label_loc, items)
    plt.title('Five Psychological Motivations', size=20)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend()
    fig.savefig('static/radar.png')
    result.update({"radar": "radar.png"})

    return result
# 'radar.png', 'vectors.html', get_text(rank), explanation
