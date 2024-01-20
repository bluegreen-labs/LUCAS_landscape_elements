
 
    #--- if requested return test metrics ----
    if args.test:
        
        for i in range(5):
            n = np.random.choice(len(test_dataset))
            
            image_vis = test_dataset[n][0].astype('uint8')
            image, gt_mask = test_dataset[n]
            
            gt_mask = gt_mask.squeeze()
            
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
             = best_model.predict(x_tensor)
            
            # run test dataset
            test_metrics = trainer.test(
                model,
                dataloaders = test_dataloader,
                verbose = False
            )
            
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
                
            visualize(
                image=image_vis, 
                ground_truth_mask=gt_mask, 
                predicted_mask=pr_mask
            )

        
        print(test_metrics)

