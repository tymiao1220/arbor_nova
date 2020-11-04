export default async (girderRest) => {
  let slurmSettings;
  girderRest.user = (await girderRest.get('user/me')).data;
  if (!girderRest.user) {
    await girderRest.login('anonymous', 'letmein');
  }
  slurmSettings = (await girderRest.get('/slurm/slurmOption', {})).data;
  console.log(slurmSettings)
  return slurmSettings;
}
