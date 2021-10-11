
def is_intersect(box1, box2):
    len_x = abs((box1[0] + box1[2]) / 2 - (box2[0] + box2[2]) / 2)
    len_y = abs((box1[1] + box1[3]) / 2 - (box2[1] + box2[3]) / 2)
    box1_x = abs(box1[0] - box1[2])
    box2_x = abs(box2[0] - box2[2])
    box1_y = abs(box1[1] - box1[3])
    box2_y = abs(box2[1] - box2[3])
    if len_x <= (box1_x + box2_x) / 2 and len_y <= (box1_y + box2_y) / 2:
        return True
    else:
        return False


def compute_iou(box1, box2):
    if not is_intersect(box1, box2):
        return 0
    col = min(box1[2], box2[2]) - max(box1[0], box2[0])
    row = min(box1[3], box2[3]) - max(box1[1], box2[1])

    intersection = col * row
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    coincide = intersection / (box1_area + box2_area - intersection)
    return coincide
