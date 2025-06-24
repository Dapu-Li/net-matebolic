// .vitepress/theme/index.ts
import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import CsvTable from './components/CsvTable.vue'
import CsvTableIMP from './components/CsvTableIMP.vue'
import CsvTableROC from './components/CsvTableROC.vue'
import CsvTableMRF from './components/CsvTableMRF.vue'
import CsvTableMRI from './components/CsvTableMRI.vue'
import CsvTableNumb from './components/CsvTableNumb.vue'
//import CsvTable_Sen from './components/CsvTable_Sen.vue'
import './style.css'
  
export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // 可以在这里插入自定义布局插槽（例如 header/footer）
    })
  },
  enhanceApp({ app }) {
    app.component('CsvTable', CsvTable)       //  注册 CsvTable 组件
    app.component('CsvTableIMP', CsvTableIMP) //  注册 CsvTableIMP 组件
    app.component('CsvTableROC', CsvTableROC) //  注册 CsvTableROC 组件
    app.component('CsvTableMRF', CsvTableMRF) //  注册 CsvTableMR_F 组件
    app.component('CsvTableMRI', CsvTableMRI) //  注册 CsvTableMR_I 组件
    //app.component('CsvTable_Sen', CsvTable_Sen) //  注册 CsvTable_Sen 组件
    app.component('CsvTableNumb', CsvTableNumb) //  注册 CsvTableNumb 组件
  }
} satisfies Theme
