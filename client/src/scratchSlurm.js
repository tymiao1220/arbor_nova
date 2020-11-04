export default async (girderRest) => {
  let folder;
  girderRest.user = (await girderRest.get('user/me')).data;
  if (!girderRest.user) {
    await girderRest.login('anonymous', 'letmein');
  }
  slurmSetting = (await girderRest.get('/slurmOption', {})).data[0];
  return slurmSetting;
}
