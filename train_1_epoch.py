def train_1_epoch(env, obj, args, loaded_net, scheduler, optimizer):

    dt_loader = get_offline_dataset(args)

    num_epochs = 500
    losses = []
    rewards = []
    best_succ_rate = 0

    if args.cuda:
        student_model = student_model.cuda(MPI.COMM_WORLD.Get_rank())
        if not args.scripted:
            state_based_model = state_based_model.cuda(MPI.COMM_WORLD.Get_rank())

    student_model.train()
    # plot_model_stats(obj)
    rand_i = str(np.random.uniform(0,1))
    print(f"start training for {rand_i}")
    for idx, dt in enumerate(dt_loader):

        dt["obj"] = obj
        obs_state = dt["observation"]
        g = dt["desired_goal"].to(torch.float32)

        # TODO normalize
        obs_img, g_norm, state_based_input = _preproc_inputs_image_goal(dt, args, is_np=False)

        # print(obs_img.shape)
        # obs_img = obs_img.permute(0,2,3,1).numpy()[0].astype(np.uint8)
        # show_video(obs_img)

        if args.scripted:
            with torch.no_grad():
                acts = dt["actions"].clone().detach()
                if args.cuda:
                    acts = acts.cuda(MPI.COMM_WORLD.Get_rank())
        else:
            with torch.no_grad():
                acts = loaded_net(state_based_input)


        if args.task != 'sym_state':
            student_acts = loaded_net(obs_img, g_norm)
        else:
            student_acts = loaded_net(state_based_input)
        # compute the loss
        loss = F.mse_loss(student_acts, acts)

        # step the loss
        optimizer.zero_grad()
        loss.backward()
        # plot_grad_flow(state_based_model.named_parameters())
        optimizer.step()

        total_loss += loss.item()

        # TODO: add plotting for training
        losses.append(loss.item())

        scheduler.step(total_loss)

        end = time.time()

        # run after every epoch

        print(f"Epoch {ep} | Total time taken {end - start} | Loss {total_loss / len(dt_loader)}")


        if ep % 10 == 0:
            # save video of agent
            args.record = True
        else:
            args.record = False

        succ_rate = eval_agent_and_save(ep, env, args, student_model, obj, task=args.task)
        #succ_rate = 0
        rewards.append(succ_rate)
        save_dict = {
            'actor_net': student_model.state_dict(),
            'o_mean': obj["o_mean"],
            'o_std': obj["o_std"],
            'g_mean': obj["g_mean"],
            'g_std': obj["g_std"],
            'reward_plots': rewards,
            'losses': losses,
        }
        if succ_rate >= best_succ_rate:
            torch.save(save_dict, f"best_bc_model_{rand_i}.pt")
            best_succ_rate = succ_rate
        else:
            torch.save(save_dict, f"curr_bc_model_{rand_i}.pt")

    return loaded_model, scheduler, optimizer
