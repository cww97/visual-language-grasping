import sys
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator(sys.argv[1], size_guidance={"scalars":1000000})
ea.Reload()

success = ea.scalars.Items('reactive/grasp_success')
label = ea.scalars.Items('reactive/label')

SUCCESS = 0
FAIL = 1
WRONG = 2
total = 0
grasp = 0
succ = 0
succcnt = []
graspcnt = []
previntruction = None
cnt = 0
now = 0
for x, y in zip(success[int(sys.argv[3]):], label[int(sys.argv[3]):]):
    # print(x.step, x.value, y.value)
    instruction = open(sys.argv[2] + '/data/instructions/' + '%06d.0.instruction.txt' % (x.step)).read().strip()
    print(instruction, x.value)
    if instruction != previntruction:
        cnt = 0
        now = 0
    previntruction = instruction
    total += 1
    cnt += 1
    now += 1
    if x.value == SUCCESS:
        succ += 1
        grasp += 1
        succcnt.append(cnt)
        graspcnt.append(now)
        now = 0
        cnt = 0
    elif x.value == WRONG:
        grasp += 1
        graspcnt.append(now)
        now = 0

print(succ, grasp)
print("total: ", total)
print("success %: ", grasp / total * 100)
print("correct %: ", succ / grasp * 100)

print("average success count: ", sum(graspcnt) / len(graspcnt))
print("average correct count: ", sum(succcnt) / len(succcnt))
# print(len(success))