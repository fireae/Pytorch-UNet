from utils import get_batch_images_masks

if __name__ == '__main__':
    data_iter = get_batch_images_masks('data/images', 'data/masks')
    for i in range(2):
        out = next(data_iter)
        # print(out)