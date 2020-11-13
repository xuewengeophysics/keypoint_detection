import torch

def pkd_loss(student_atts, teacher_atts, student_reps, teacher_reps, device='cuda'):
    teacher_atts = [teacher_atts[i] for i in [1, 3, 5, 7, 9]]
    att_tmp_loss, rep_tmp_loss = [], []
    for student_att, teacher_att in zip(student_atts, teacher_atts):
        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                    student_att)
        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                    teacher_att)

        att_tmp_loss.append(torch.nn.functional.mse_loss(student_att, teacher_att))
    att_loss = sum(att_tmp_loss)
    new_teacher_reps = [teacher_reps[i] for i in [2, 4, 6, 8, 10]]
    new_student_reps = student_reps[1:-1]
    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
        rep_tmp_loss.append(torch.nn.functional.mse_loss(student_rep, teacher_rep))
    rep_loss = sum(rep_tmp_loss)

    return att_loss, rep_loss