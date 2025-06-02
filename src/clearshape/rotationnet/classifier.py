import torch.nn as nn

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")
    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet' :
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
    

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        nsamp = int( input.size(0) / nview )

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_ = torch.LongTensor( target.size(0) * nview )

        # compute output
        output = model(input_var)
        num_classes = int( output.size( 1 ) / nview ) - 1
        output = output.view( -1, num_classes + 1 )

        ###########################################
        # compute scores and decide target labels #
        ###########################################
        output_ = torch.nn.functional.log_softmax( output )
        # divide object scores by the scores for "incorrect view label" (see Eq.(5))
        output_ = output_[ :, :-1 ] - torch.t( output_[ :, -1 ].repeat( 1, output_.size(1)-1 ).view( output_.size(1)-1, -1 ) )
        # reshape output matrix
        output_ = output_.view( -1, nview * nview, num_classes )
        output_ = output_.data.cpu().numpy()
        output_ = output_.transpose( 1, 2, 0 )
        # initialize target labels with "incorrect view label"
        for j in range(target_.size(0)):
            target_[ j ] = num_classes
        # compute scores for all the candidate poses (see Eq.(5))
        scores = np.zeros( ( vcand.shape[ 0 ], num_classes, nsamp ) )
        for j in range(vcand.shape[0]):
            for k in range(vcand.shape[1]):
                scores[ j ] = scores[ j ] + output_[ vcand[ j ][ k ] * nview + k ]
        # for each sample #n, determine the best pose that maximizes the score for the target class (see Eq.(2))
        for n in range( nsamp ):
            j_max = np.argmax( scores[ :, target[ n * nview ], n ] )
            # assign target labels
            for k in range(vcand.shape[1]):
                target_[ n * nview * nview + vcand[ j_max ][ k ] * nview + k ] = target[ n * nview ]
        ###########################################

        target_ = target_.cuda()
        target_var = torch.autograd.Variable(target_)

        # compute loss
        loss = criterion(output, target_var)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
