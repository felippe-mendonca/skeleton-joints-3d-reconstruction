import cv2
from itertools import permutations

COLORS = list(permutations([0, 255, 85, 170], 3))


def draw_skeletons(image, skeletons, links):

    for ob in skeletons.objects:
        parts = {}
        for part in ob.keypoints:
            parts[part.id] = (int(part.position.x), int(part.position.y))

        for link_parts, color in zip(links, COLORS):
            begin, end = link_parts
            if begin in parts and end in parts:
                cv2.line(image, parts[begin], parts[end], color=color, thickness=4)

        for _, center in parts.items():
            radius = 3
            cv2.circle(image, center=center, radius=radius, color=(255, 255, 255), thickness=-1)

    return image