question = f"<image>\nQuestion : Do the square points higher, slightly higher, lower, slightly lower, or overlapping than the diamond point?"
pixel_values = load_image("testdb_20241020/testcase_1.png", max_num=12).to(torch.bfloat16).cuda()
generation_config  = dict(max_new_tokens=2048, do_sample=False)
downsample_ratio = 0.5
patch_size = 14
image_size = 448 #config.force_image_size or config.vision_config.image_size
num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))

IMG_START_TOKEN='<img>'; IMG_END_TOKEN='</img>'; IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
template = get_conv_template("internlm2-chat")
eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
template.append_message(template.roles[0], question)
template.append_message(template.roles[1], None)
query = template.get_prompt()
for num_patches in num_patches_list:
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
    query = query.replace('<image>', image_tokens, 1)
model_inputs = tokenizer(query, return_tensors='pt')
input_ids = model_inputs['input_ids'].to(device)
attention_mask = model_inputs['attention_mask'].to(device)

model.extract_feature(pixel_values)
