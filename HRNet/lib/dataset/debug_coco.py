from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def loadNumpyAnnotations(self, data):
    """
    Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    :param  data (numpy.ndarray)
    :return: annotations (python nested list)
    """
    print('Converting ndarray to lists...')
    assert (type(data) == np.ndarray)
    print(data.shape)
    assert (data.shape[1] == 7)
    N = data.shape[0]
    ann = []
    for i in range(N):
        if i % 1000000 == 0:
            print('{}/{}'.format(i, N))
        ann += [{
            'image_id': int(data[i, 0]),
            'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
            'score': data[i, 5],
            'category_id': int(data[i, 6]),
        }]
    return ann

if __name__ == '__main__':
    #annotation_file = '../../data/infrared/annotations/pig_keypoints_val2020.json'
    annotation_file = '../../data/coco/annotations/person_keypoints_val2017.json'
    coco = COCO(annotation_file)
    for item in coco:
        print(item)
    """
    resFile = '../../output/infrared/pose_hrnet/w32_256x192_adam_lr1e-3/results/keypoints_val2020_results_0.json'
    anns = json.load(open(resFile))
    print(type(anns)) # list
    annsImgIds = [ann['image_id'] for ann in anns]
    print("annsImgIds = ", set(annsImgIds))
    print("getImgIds = ", set(coco.getImgIds()))
    #coco_dt = coco.loadRes(resFile)
    #coco_eval = COCOeval(coco, None, 'keypoints')

    # 获得数据集中标注的类别，该处只有pig一个类
    cats = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
    print(cats)
    # 所有类别前面，加上一个背景类
    classes = ['__background__'] + cats
    # 计算包括背景所有类别的总数
    num_classes = len(classes)
    # 字典  类别名:类别顺序编号
    _class_to_ind = dict(zip(classes, range(num_classes)))
    print(_class_to_ind)
    # 字典  类别标签编号:coco数据类别编号
    _class_to_coco_ind = dict(zip(cats, coco.getCatIds()))
    print(_class_to_coco_ind)
    # 字典  coco数据类别编号:类别顺序编号
    _coco_ind_to_class_ind = dict(
        [
            (_class_to_coco_ind[cls], _class_to_ind[cls])
            for cls in classes[1:]
        ]
    )
    print(_coco_ind_to_class_ind)
    # 获得包含person图象的标号，即image_id
    image_set_index = coco.getImgIds()
    print("image_set_index = ", image_set_index)
    """